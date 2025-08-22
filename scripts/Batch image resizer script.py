import pathlib # Manejo moderno de rutas de archivos y directorios
import cv2 # OpenCV: lectura de imágenes

# VARIABLES
extensiones_lista = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif', 'heic'] # Lista de formatos de imagen soportados por OpenCV
valor_usuario = 0
directorio_entrada = ""

# FUNCIONES
def procesar_directorio_imagenes():
    global extensiones_lista, valor_usuario, directorio_entrada, ancho_val, alto_val
    
    # Generar nombre para directorio de salida
    carpeta_salida = pathlib.Path(directorio_entrada).parent / f"{pathlib.Path(directorio_entrada).name} (output)"

    # Crear carpeta de destino si no existe
    carpeta_salida.mkdir(parents = True, exist_ok = True)
    
    # Mostrar separador visual para inicio de resultados
    print("-" * 36)

    cont_archivos = 0 # Contador de archivos generados
    
    # Procesar recursivamente cada imagen en el directorio
    for extension_val in extensiones_lista:
        for archivo_dir in pathlib.Path(directorio_entrada).rglob(f'*.{extension_val}'):
            if archivo_dir.is_file():
                # Cargar imagen desde archivo usando OpenCV
                imagen_val = cv2.imread(str(archivo_dir))
                
                # Capturar dimensiones de la imagen
                alto_val, ancho_val = imagen_val.shape[:2]
                
                lado_minimo = 0
                
                # Determinar el lado menor
                if ancho_val < alto_val:
                    lado_minimo = ancho_val
                else:
                    lado_minimo = alto_val
                
                # Establecer nuevo ancho y alto para imagen de salida
                ancho_val = round(ancho_val * (valor_usuario / lado_minimo))
                alto_val = round(alto_val * (valor_usuario / lado_minimo))
                
                # Redimensionar imagen a nuevos valores de ancho y alto
                imagen_val = cv2.resize(imagen_val, (ancho_val, alto_val))
                
                # Convertir imagen a JPG
                ok_val, imagen_val = cv2.imencode(".jpg", imagen_val, [cv2.IMWRITE_JPEG_QUALITY, 100])
                imagen_val = cv2.imdecode(imagen_val, cv2.IMREAD_UNCHANGED) # Decodificar el buffer de memoria para volver a obtener la imagen ya comprimida en formato OpenCV

                # Ruta relativa
                ruta_relativa = archivo_dir.parent.relative_to(directorio_entrada)

                # Generar ruta completa de archivo de salida
                directorio_salida = carpeta_salida / ruta_relativa
                
                # Crear directorio para guardar imagen
                directorio_salida.mkdir(parents = True, exist_ok = True)
                
                # Guardar imágen en directorio de salida correspondiente
                cv2.imwrite(str(directorio_salida / archivo_dir.with_suffix(".jpg").name), imagen_val)
                
                # Mostrar archivo procesado
                print(str(archivo_dir))

                cont_archivos += 1 # Aumentar el contador de archivos

    # En caso de no procesar ninguna imagen
    if cont_archivos == 0:
        print("No images processed")
    
    # Mostrar separador final
    print("-" * 36 + "\n")

# PUNTO DE PARTIDA
# Bucle principal del programa
while True:
    # Solicitar directorio de entrada
    while True:
        directorio_entrada = input("Enter directory: ").strip('"\'')
        
        # Verificar que el directorio exista
        if not pathlib.Path(directorio_entrada).exists():
            print("Wrong directory\n")
        else:
            break
    
    # Solicitar lado mínimo de imagen
    while True:
        valor_usuario = input("Enter minimum size for width or height: ")
        
        # Únicamente número
        if valor_usuario.isdigit():
            valor_usuario = int(valor_usuario)
            
            break
        else:
            print("Wrong format\n")
    
    # Procesar directorio de imágenes recursivamente
    procesar_directorio_imagenes()