import os
import re
import sys
import time
import unicodedata
import logging
from pathlib import Path

import httpx
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Sección de Configuración:
#
# Obtener las variables de entorno (path de la carpeta objetivo y API keys 
# del servicio de transcripción y procesamiento de texto) desde un archivo 
# .env en la misma carpeta que el script. Si alguna falta, finaliza la
# ejecución con un mensaje de error.
#
# Se definen constantes de utilidad como el TIMEOUT, el modelo de Gemini,
# los reintentos máximos, la espera base entre intentos, la configuración
# de Deepgram, el prompt para pulir la transcripción y las extendiones de
# audio soportadas.
#
# ---------------------------------------------------------------------------


load_dotenv()

def obtener_env(nombre: str) -> str:
    valor = os.environ.get(nombre)
    if not valor:
        sys.exit(f"Falta la variable {nombre} en el .env")
    return valor

VAULT_YEAR_DIR = Path(obtener_env("VAULT_YEAR_DIR"))
DEEPGRAM_API_KEY = obtener_env("DEEPGRAM_API_KEY")
GEMINI_API_KEY = obtener_env("GEMINI_API_KEY")

if not VAULT_YEAR_DIR.is_dir():
    sys.exit(f"No se puede encontrar la carpeta del anio indicado: {VAULT_YEAR_DIR}")

NOMBRE_CARPETA_AUDIOS = "audios"
NOMBRE_CARPETA_APUNTES = "apuntes"

GEMINI_MODEL = "gemini-3.6-flash"

DEEPGRAM_TIMEOUT = httpx.Timeout(900.0, connect=15.0)

# Cantidad de reintentos ante fallos y espera inicial ante un fallo.
REINTENTOS_MAX = 3
ESPERA_BASE_SEGUNDOS = 15

# Parámetros de configuración para Deepgram.
DEEPGRAM_OPTIONS = PrerecordedOptions(
    model="nova-3",
    language="es",
    smart_format=True,
    punctuate=True,
    diarize=True,
    paragraphs=True,
    numerals=True,
)

PROMPT_PULIDO = """Tu trabajo será pulir la transcripción generada por Deepgram a partir de la grabación de audio de una clase de una materia.
Qué quiero:
- Que en un archivo .md perfecciones la transcripción de modo que quede lista como apunte de estudio.
- No es un resumen. La idea es perfeccionar la clase tal como fue dictada: eliminar incoherencias propias de una transcripción automática (repeticiones, muletillas, cortes de frase, marcas de oralidad tipo "¿no?", "¿sí?"), unir párrafos que traten el mismo tema y separar el contenido por temas, cada uno con su subtítulo.
- Mantené el nivel de detalle real de la clase: si el profesor da una definición, transcribila completa; si da un ejemplo relevante para entender un concepto, conservalo. No comprimas ni sintetices el contenido académico.
Al inicio del archivo agregá:
Un índice de contenidos (los temas tratados en la clase, en el orden en que aparecen)
Una sección de palabras clave
En la transcripción vas a encontrar varios "Speakers" identificados. Me interesa únicamente lo que dice el profesor/la profesora. Identificá vos, por el contenido, cuál Speaker es el docente (suele ser quien explica los conceptos, da indicaciones y lleva el hilo de la clase) — no asumas que siempre va a ser el mismo número de Speaker entre una clase y otra.
Del diálogo entre el profesor y el resto de los speakers (alumnos), no reproduzcas el ida y vuelta conversacional, pero si de ese intercambio surgió un concepto importante o una definición relevante, incorporalo al apunte redactado en forma de contenido, no como diálogo.
Eliminá todos los modismos conversacionales y marcas de oralidad. El resultado tiene que leerse como un apunte para estudiar, no como una transcripción.
Agregá, en una sección aparte y claramente diferenciada del resto del apunte, todo lo que el profesor haya dicho sobre: modalidad de cursada, sistema de evaluación (parciales, recuperatorios, condiciones de promoción), actividades prácticas sugeridas, fechas importantes, y cualquier otra cuestión organizativa que valga la pena registrar. Si en esta clase en particular no se mencionó nada de esto, omití directamente la sección (no la inventes ni la fuerces).
Adjunto la transcripción de la clase.
Esta clase puede tener contenido matemático (fórmulas, derivadas, integrales, límites,
gráficos, demostraciones). Para esas partes:
- Convertí toda la matemática hablada a notación real en formato LaTeX de Obsidian:
expresiones dentro de una oración entre $...$, y fórmulas que merezcan destacarse
aparte entre $$...$$ en su propia línea.
- Conservá el desarrollo completo de cualquier ejemplo o demostración resuelta paso a
paso: no saltees directo al resultado, mantené cada paso del razonamiento.
- Prestá especial atención a la exactitud de números, exponentes, coeficientes y
símbolos. Si una parte de la transcripción es ambigua y no podés inferir con
confianza qué número o símbolo corresponde, marcala como [verificar] en vez de
adivinar.
- Si el profesor describe un gráfico, diagrama o algo escrito en el pizarrón que no
quedó claro solo con el audio, agregá una nota entre corchetes indicando que ahí
hubo contenido visual no capturado, para que sepa que tengo que completarlo con
tus propias fotos o apuntes en papel.
- Cuando el profesor enuncie una definición, teorema o fórmula importante, destacala
  visualmente (por ejemplo en un bloque de cita con >), separada del resto del texto.
- Si eres un modelo con acceso a internet o una buena base de conocimiento, puedes utilizar ambos para completar el concepto que en la transcripción no quede claro.
""".strip()

# Múltiples extensiones de audio por si se graba en múltiples formatos.
EXTENSIONES_AUDIO = {".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac"}

# Configuración del logger. Registra tanto en consola como en un .log.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("procesar_apuntes.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sección de utilidades y funciones de apoyo:
# 
# Se definen respuestas ante errores, reintentos, extracción de números de 
# clase, construcción de nombres de archivos, listado de cátedras y 
# clases pendientes, y priorización de procesamiento.
#
# ---------------------------------------------------------------------------


#  Excepción que indica exceso de uso de Gemini.
class CuotaAgotada(Exception):
    pass

# Controla la ejecución de los reintentos ante fallos de Deepgram o 
# Gemini. Si se agota la cuota de Gemini, lanza CuotaAgotada.
def con_reintentos(func, descripcion: str):
    for intento in range(1, REINTENTOS_MAX + 1):
        try:
            return func()
        except CuotaAgotada:
            raise
        except Exception as e:
            if intento == REINTENTOS_MAX:
                raise
            espera = ESPERA_BASE_SEGUNDOS * intento
            log.warning(
                "  %s: intento %d/%d fallo (%s). Reintento en %ds...",
                descripcion, intento, REINTENTOS_MAX, e, espera,
            )
            time.sleep(espera)

# Extrae el número de clase del nombre de un archivo utilzando una expresión regular.
# Si no se puede interpretar el nombre del archivo, devuelve None.
def extraer_numero_clase(nombre_archivo: str):
    m = re.search(r"clase[_\s]*(\d+)", Path(nombre_archivo).stem, re.IGNORECASE)
    return int(m.group(1)) if m else None

# Crea el nombre del apunte con PascalCase y sin tildes, siguiendo el formato Materia_Clase_N.md.
def construir_nombre_apunte(materia: str, n: int) -> str:
    palabras = re.split(r"\s+", materia.strip())
    pascal = []
    for palabra in palabras:
        sin_tildes = unicodedata.normalize("NFKD", palabra)
        sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
        pascal.append(sin_tildes.capitalize())
    return f"{''.join(pascal)}_Clase_{n}.md"



# ---------------------------------------------------------------------------
# Sección de funciones principales:
#
# Se listan las cátedras haciendo un escaneo de las subcarpetas en el
# año indicado. Se compara la lista de audios con la de apuntes para ver
# qué clases no fueron procesadas y se permite priorizar el procesamiento 
# de alguna cátedra en particular. Luego se procesan las demás clases en
# orden, guardando el resultado en la carpeta de apuntes correspondiente.
#
# ---------------------------------------------------------------------------


# Escanea la carpeta del año indicado y devuelve una lista de cátedras con
# subcarpetas de audios y apuntes. En caso de no tener, la salta.
def listar_catedras():
    catedras = []
    for carpeta in sorted(VAULT_YEAR_DIR.iterdir()):
        if not carpeta.is_dir():
            continue
        audios_dir = carpeta / NOMBRE_CARPETA_AUDIOS
        apuntes_dir = carpeta / NOMBRE_CARPETA_APUNTES
        if audios_dir.is_dir() and apuntes_dir.is_dir():
            catedras.append({
                "materia": carpeta.name,
                "audios_dir": audios_dir,
                "apuntes_dir": apuntes_dir,
            })
        else:
            log.debug("La carpeta '%s' no tiene audios/ ni apuntes/. Son necesarias para el funcionamiento del script.", carpeta.name)
    return catedras

# Permite indicar una cátedra para priorizar su procesamiento. Se puede 
# ignorar y procesar en orden de aparicion. Retorna None o el nombre
# de la cátedra prioritaria.
def elegir_prioridad(pendientes: list):
    materias = list(dict.fromkeys(a["materia"] for a in pendientes))  # sin duplicados, en orden
    if len(materias) <= 1:
        return None

    print("\nCatedras con clases pendientes:")
    for i, materia in enumerate(materias, start=1):
        cantidad = sum(1 for a in pendientes if a["materia"] == materia)
        print(f"  {i}. {materia} ({cantidad} clase(s))")

    respuesta = input(
        "\nPriorizar alguna? Numero de la lista, o Enter para procesar en el orden de arriba: "
    ).strip()
    if not respuesta:
        return None
    try:
        indice = int(respuesta) - 1
        if 0 <= indice < len(materias):
            return materias[indice]
    except ValueError:
        pass
    print("No se ingresó una opción válida. Se procesará en el orden de aparición.")
    return None

# En caso de priorizar una cátedra, reordena la lista de pendientes poniendo primero
# las clases de la cátedra prioritaria, sin alterar el orden del resto.
def reordenar_por_prioridad(pendientes: list, materia_prioritaria: str) -> list:
    prioritarias = [a for a in pendientes if a["materia"] == materia_prioritaria]
    resto = [a for a in pendientes if a["materia"] != materia_prioritaria]
    return prioritarias + resto

# Función encargada de encontrar las clases pendientes de procesar. Compara los 
# archivos de audio con los apuntes ya generados y devuelve una lista de diccionarios 
# con la información de cada clase pendiente.
def encontrar_pendientes() -> list:
    pendientes = []
    for catedra in listar_catedras():
        clases_listas = set()
        for archivo in catedra["apuntes_dir"].glob("*.md"):
            n = extraer_numero_clase(archivo.name)
            if n is not None:
                clases_listas.add(n)

        for archivo in sorted(catedra["audios_dir"].iterdir()):
            if archivo.suffix.lower() not in EXTENSIONES_AUDIO:
                continue
            n = extraer_numero_clase(archivo.name)
            if n is None:
                log.warning(
                    "No pude interpretar el numero de clase en '%s' (%s), lo salteo.",
                    archivo.name, catedra["materia"],
                )
                continue
            if n not in clases_listas:
                pendientes.append({
                    "materia": catedra["materia"],
                    "n": n,
                    "path": archivo,
                    "apuntes_dir": catedra["apuntes_dir"],
                })
    # Lambda devuelve tuplas (materia, n) para ordenar primero por materia y luego por número de clase.
    pendientes.sort(key=lambda a: (a["materia"], a["n"]))
    return pendientes


# ---------------------------------------------------------------------------
# Sección de funciones de procesamiento (pipeline):
#
# Se definen las funciones que realizan la transcripción de audio a texto.
# Se invoca a Deepgram pasándole los parámetros de configuración y el archivo
# de audio. Con la respuesta de deepgram, y junto con el promt de pulido, 
# se invoca a Gemini para pulir el apunte. Gemini devuelve un .md que se
# guarda en la subcarpeta de apuntes correspondiente.
#
# ---------------------------------------------------------------------------

# Se abre el archivo de audio en binario y luego se usa transcribe_file
# pasándole el binario del audio, los parámetros y el timeout. Si es exitoso,
# devuelve la transcripción cruda.
def transcribir(path: Path) -> str:
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    with open(path, "rb") as audio_file:
        buffer_data = audio_file.read()
    response = deepgram.listen.rest.v("1").transcribe_file(
        source={"buffer": buffer_data},
        options=DEEPGRAM_OPTIONS,
        timeout=DEEPGRAM_TIMEOUT,
    )
    return response.results.channels[0].alternatives[0].transcript

# Se invoca a Gemini pasándole la trancripción cruda y el promt de pulido.
# Si es exitoso, devuelve el contenido en formato .md listo para guardar.
# También verifica si se agotó la cuota de uso de Gemini y en tal caso
# lanza la excepción CuotaAgotada.
def pulir(transcripcion: str, materia: str, n: int) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    contenido = (
        f"Materia: {materia}\nClase numero: {n}\n\nTranscripcion cruda:\n{transcripcion}"
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contenido,
            config=types.GenerateContentConfig(
                system_instruction=PROMPT_PULIDO,
                temperature=0.3,
            ),
        )
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) and "PerDay" in str(e):
            raise CuotaAgotada(str(e)) from e
        raise
    return response.text

# Función principal. Recibe un diccionario con información de la clase a procesar.
# Invoca a transcribir() y luego a pulir(), y guarda el resultado en la carpeta de
# apuntes correspondiente invocando con el nombre creado por construir_nombre_apunte().
# Registra en el log el inicio y fin del procesamiento.
def procesar_clase(item: dict) -> None:
    materia, n, path = item["materia"], item["n"], item["path"]
    log.info("Procesando: %s - Clase %s", materia, n)
    transcripcion = con_reintentos(lambda: transcribir(path), "Deepgram")
    log.info("  Transcripcion lista (%d caracteres)", len(transcripcion))
    markdown = con_reintentos(lambda: pulir(transcripcion, materia, n), "Gemini")
    destino = item["apuntes_dir"] / construir_nombre_apunte(materia, n)
    destino.write_text(markdown, encoding="utf-8")
    log.info("  Guardado: %s", destino)


# ---------------------------------------------------------------------------
# MAIN:
#
# Se buscan las clases pendientes, se permite priorizar una cátedra, y se
# procesan todas las clases pendientes en orden, registrando el resultado en el log.
#
# ---------------------------------------------------------------------------

def main():
    pendientes = encontrar_pendientes()

    if not pendientes:
        log.info("No hay clases pendientes.")
        return

    print(f"\Hay {len(pendientes)} clase(s) pendiente(s):")
    for a in pendientes:
        print(f"  - {a['materia']} - Clase {a['n']}")
    if len(pendientes) > 15:
        print(f"\n(Se pueden procesar al rededor de 20 clases por día. Hay {len(pendientes)}, por lo que puede que no se completen todas.")

    prioridad = elegir_prioridad(pendientes)
    if prioridad:
        pendientes = reordenar_por_prioridad(pendientes, prioridad)
        print(f"\nSe reordenó la lista. Inicia con {prioridad}:")
        for a in pendientes:
            print(f"  - {a['materia']} - Clase {a['n']}")

    respuesta = input("\nProcesar todas? [s/n]: ").strip().lower()
    if respuesta != "s":
        print("Cancelado, no se proceso nada.")
        return

    exitosas, fallidas = 0, 0
    for item in pendientes:
        try:
            procesar_clase(item)
            exitosas += 1
        except CuotaAgotada:
            # Loguear el error, se consumió la cuota.
            log.error("Se consumio la cuota diaria de Gemini.")
            # Informar en consola que se terminó la cuota diaria.
            print(f"\nSe consumio la cuota diaria de Gemini.")
            print(f"Se procesaron {exitosas} de {len(pendientes)}.")
            print("Toda clase no procesada queda como pendiente")
            return
        except Exception:
            log.exception("  Fallo procesando %s - Clase %s (se agotaron los reintentos)", item["materia"], item["n"])
            fallidas += 1

    print(f"\nFinalizado. Se procesaron exitosamente {exitosas} clases. {fallidas} clases fallaron.")
    if fallidas:
        print("Revisar el log de errores: procesar_apuntes.log para más detalles.")


if __name__ == "__main__":
    main()
