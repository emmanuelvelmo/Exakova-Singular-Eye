import shutil
import pathlib
import collections

# Generar un nombre de directorio único para evitar sobreescribir
def f_directorio_salida():
    contador_val = 0
    directorio_salida = ""

    while True:
        if contador_val == 0:
            directorio_salida = f"Output files"
        else:
            directorio_salida = f"Output files ({contador_val})"
        
        if not pathlib.Path(directorio_salida).exists():
            break
        
        contador_val += 1

    return directorio_salida

# Agrupar archivos por carpeta
def f_agrupar_por_carpeta(directorio_capt, extension_val):
    archivos_por_carpeta = collections.defaultdict(list)

    for entrada in pathlib.Path(directorio_capt).rglob(f'*.{extension_val}'):
        if entrada.is_file():
            archivos_por_carpeta[entrada.parent].append(entrada)

    return archivos_por_carpeta

# Copiar y renombrar archivos en cada carpeta con su propio contador
def f_copiar_archivos(directorio_capt, archivos_por_carpeta, directorio_salida, extension_val):
    total_archivos = 0

    for carpeta_origen, archivos in archivos_por_carpeta.items():
        cantidad = len(archivos)
        ancho_numeracion = len(str(cantidad))

        # Ruta relativa de carpeta origen respecto al directorio base
        carpeta_destino = pathlib.Path(directorio_salida) / carpeta_origen.relative_to(directorio_capt)

        # Crear estructura de carpetas destino
        carpeta_destino.mkdir(parents=True, exist_ok=True)

        # Iterar sobre los archivos
        for i in range(cantidad):
            nombre_archivo = f"{str(i + 1).zfill(ancho_numeracion)}.{extension_val}"
            shutil.copy2(archivos[i], carpeta_destino / nombre_archivo)
            
            total_archivos += 1

    return total_archivos

# Bucle principal continuo
while True:
    # Directorio de entrada
    while True:
        directorio_capt = input("Enter directory: ").strip('"\'')

        if pathlib.Path(directorio_capt).exists():
            break

        print("Wrong directory")

    extension_val = input("Enter file extension: ").strip('"\'')

    print("------------------------------------")

    # Agrupar archivos por carpeta
    archivos_por_carpeta = f_agrupar_por_carpeta(directorio_capt, extension_val)

    # Obtener directorio de salida
    directorio_salida = f_directorio_salida()

    # Crear directorio de salida
    pathlib.Path(directorio_salida).mkdir()

    # Contar, copiar y renombrar archivos
    contador_archivos = f_copiar_archivos(directorio_capt, archivos_por_carpeta, directorio_salida, extension_val)

    if contador_archivos == 0:
        # Eliminar directorio
        shutil.rmtree(directorio_salida)
        
        print("No modified files")
    elif contador_archivos == 1:
        print("1 modified file")
    else:
        print(f"{contador_archivos} modified files")

    print("------------------------------------\n")