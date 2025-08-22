import os # Para operaciones del sistema de archivos
import time # Para pausas y delays
import re # Para expresiones regulares
import requests # Para realizar peticiones HTTP
import urllib.parse # Para codificar URLs
import selenium.webdriver # Para automatización del navegador web
from selenium.webdriver.common.by import By # Para localizar elementos en la página
from selenium.webdriver.chrome.service import Service # Para el servicio de ChromeDriver
from selenium.webdriver.chrome.options import Options # Para configurar opciones de Chrome

# Bucle principal del programa
while True:
    # Solicitar descripción de imagen y cantidad
    texto_buscar = input("Image description: ").strip('"\'')
    
    num_imgs = 0 # 
    
    # Solicitar lado mínimo de imagen
    while True:
        num_imgs = input("Number of images: ")
        
        # Únicamente número
        if num_imgs.isdigit():
            num_imgs = int(num_imgs)
            
            break
        else:
            print("Wrong format")
    
    # Mostrar separador visual para inicio de resultados
    print("\n" + "-" * 36 + "\n" + f"Progress: 0/{num_imgs}", end = "\r")
    
    # Crear carpeta de salida
    if not os.path.exists(texto_buscar):
        os.makedirs(texto_buscar)
    
    # Configuración de Selenium
    opciones_chrome = Options()
    
    opciones_chrome.add_argument("--headless") # corre sin ventana
    opciones_chrome.add_argument("--disable-gpu") # Desactiva la aceleración GPU
    opciones_chrome.add_argument("--log-level=3") # Reduce mensajes de log
    opciones_chrome.add_argument("--window-size=1280,720") # Define tamaño de ventana
    opciones_chrome.add_argument("--no-sandbox") # Mejora compatibilidad
    opciones_chrome.add_argument("--disable-dev-shm-usage") # Evita problemas de memoria
    opciones_chrome.add_argument("--disable-extensions") # Desactiva extensiones
    opciones_chrome.add_argument("--disable-logging") # Desactiva logging adicional
    opciones_chrome.add_argument("--silent") # Modo silencioso
    opciones_chrome.add_argument("--disable-web-security") # Desactiva seguridad web
    opciones_chrome.add_argument("--remote-debugging-port=0") # Desactiva puerto de depuración remota
    
    # Suprimir logs adicionales del servicio
    servicio_val = Service()
    
    servicio_val.creation_flags = 0x08000000 # CREATE_NO_WINDOW en Windows
    
    # Inicializar el driver de Chrome
    drivesolicitud_val = selenium.webdriver.Chrome(service = servicio_val, options = opciones_chrome)
    
    # Generar URL de búsqueda
    # Codificar el texto para URL
    texto_codificado = urllib.parse.quote(texto_buscar)
    
    # Construir URL de búsqueda de Pinterest
    url_busqueda = f"https://www.pinterest.com/search/pins/?q={texto_codificado}&rs=typed"
    
    # Navegar a la página de búsqueda
    drivesolicitud_val.get(url_busqueda)
    
    # Extraer enlaces de imágenes en página cargada
    enlaces_val = set() # Usar set para evitar duplicados
    
    pausa_scroll = 2 # Tiempo de pausa entre scrolls
    
    altura_anterior = drivesolicitud_val.execute_script("return document.body.scrollHeight") # Obtener altura inicial de la página
    
    patron_img = re.compile(r"/(\d{3,})x/") # busca carpetas con formato mayor a 200 seguido x
    
    # Contador de intentos
    intentos_scroll = 0
    maximo_intentos = 3 # Máximo de intentos sin encontrar nuevo contenido
    
    # Buscar imágenes haciendo scroll hasta encontrar la cantidad deseada
    while len(enlaces_val) < num_imgs and intentos_scroll < maximo_intentos:
        # Encontrar todas las imágenes en la página actual
        imgs_val = drivesolicitud_val.find_elements(By.TAG_NAME, "img")
        
        # Procesar cada imagen encontrada
        for iter_img in imgs_val:
            src_val = iter_img.get_attribute("src") # Obtener URL de la imagen
            
            # Filtrar solo imágenes
            if src_val and any(src_val.lower().endswith(ext_val) for ext_val in ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.svg')) and patron_img.search(src_val):                # solo contar si es nuevo
                    if src_val not in enlaces_val:
                        enlaces_val.add(src_val) # Agregar al conjunto de enlaces
                        
                        # Mostrar progreso
                        print(f"Progress: {len(enlaces_val)}/{num_imgs}", end = "\r")
            
            # Se llega al número de imágenes solicitadas
            if len(enlaces_val) >= num_imgs:
                break
        
        # Hacer scroll hacia abajo
        drivesolicitud_val.execute_script("window.scrollTo(0, document.body.scrollHeight);") # Scroll al final de la página
        
        time.sleep(pausa_scroll) # Pausa para cargar contenido
        
        # Revisar si llegamos al final
        nueva_altura = drivesolicitud_val.execute_script("return document.body.scrollHeight") # Nueva altura después del scroll
        
        # Si la altura no cambió, incrementar el contador de intentos
        if nueva_altura == altura_anterior:
            intentos_scroll += 1
        else:
            intentos_scroll = 0 # Reiniciar contador si hay nuevo contenido
        
        # Actualizar altura anterior
        altura_anterior = nueva_altura
    
    # Cerrar el navegador
    drivesolicitud_val.quit()
    
    print("\n") # Salto doble de línea
    
    # Descargar imágenes
    cont_img = 0 # Contador de imágenes descargadas exitosamente
   
    # Iterar sobre los enlaces y descargar cada imagen
    for pos_val, url_iter in enumerate(list(enlaces_val)[:num_imgs], 1):
        try:
            solicitud_val = requests.get(url_iter, timeout = 10) # Realizar petición HTTP con timeout
            
            # Verificar si la descarga fue exitosa
            if solicitud_val.status_code == 200:
                # Extraer el nombre original de la URL
                nombre_imagen = os.path.basename(urllib.parse.urlparse(url_iter).path)

                directorio_salida = os.path.join(texto_buscar, nombre_imagen)
                
                # Guardar imagen
                with open(directorio_salida, "wb") as f_val:
                    f_val.write(solicitud_val.content) 
                    
                cont_img += 1 # Incrementar contador de éxitos
                
                # Mostrar progreso de descarga
                print(f"{pos_val}: {nombre_imagen}")
        except Exception as e:
            continue # Continuar con la siguiente imagen si hay error
    
    # Mostrar separador final
    print("-" * 36 + "\n")