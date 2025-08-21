import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import json

class MultiOutputImageClassifier:
    def __init__(self, dataset_path="dataset", img_size=(256, 256)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.model = None
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
        """Carga y preprocesa las imágenes del dataset"""
        images = []
        labels = {'body_type': [], 'gender': [], 'skin_color': []}
        image_files = []
        
        print("Cargando imágenes del dataset...")
        
        # Primero, recopilar todas las imágenes únicas
        all_images = set()
        
        for category in ['body_type', 'gender', 'skin_color']:
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.exists(category_path):
                print(f"Advertencia: No se encontró la carpeta {category_path}")
                continue
                
            for label in os.listdir(category_path):
                label_path = os.path.join(category_path, label)
                if not os.path.isdir(label_path):
                    continue
                    
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(label_path, img_file)
                        all_images.add((img_path, category, label))
        
        # Organizar datos por imagen
        image_data = {}
        
        for img_path, category, label in all_images:
            if img_path not in image_data:
                image_data[img_path] = {'body_type': None, 'gender': None, 'skin_color': None}
            image_data[img_path][category] = label
        
        # Filtrar imágenes que tienen todas las etiquetas
        complete_images = []
        complete_labels = {'body_type': [], 'gender': [], 'skin_color': []}
        
        for img_path, img_labels in image_data.items():
            if all(img_labels[cat] is not None for cat in ['body_type', 'gender', 'skin_color']):
                # Cargar imagen
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    
                    complete_images.append(img)
                    for category in ['body_type', 'gender', 'skin_color']:
                        complete_labels[category].append(img_labels[category])
                    
                    if len(complete_images) % 100 == 0:
                        print(f"Procesadas {len(complete_images)} imágenes...")
        
        print(f"Total de imágenes cargadas: {len(complete_images)}")
        
        # Convertir a arrays numpy y normalizar
        X = np.array(complete_images, dtype=np.float32) / 255.0
        
        # Codificar etiquetas
        y_encoded = {}
        for category in complete_labels.keys():
            le = LabelEncoder()
            y_encoded[category] = le.fit_transform(complete_labels[category])
            self.label_encoders[category] = le
            print(f"{category}: {le.classes_}")
        
        return X, y_encoded
    
    def create_multi_output_model(self, num_classes_dict):
        """Crea un modelo con múltiples salidas"""
        # Entrada
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Backbone CNN compartido
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Características compartidas
        shared_features = layers.Dense(512, activation='relu')(x)
        shared_features = layers.BatchNormalization()(shared_features)
        shared_features = layers.Dropout(0.5)(shared_features)
        
        # Salidas específicas para cada categoría
        outputs = {}
        for category, num_classes in num_classes_dict.items():
            # Capas específicas para cada categoría
            category_features = layers.Dense(256, activation='relu', name=f'{category}_features')(shared_features)
            category_features = layers.Dropout(0.5)(category_features)
            
            # Salida final
            output = layers.Dense(
                num_classes, 
                activation='softmax', 
                name=f'{category}_output'
            )(category_features)
            
            outputs[category] = output
        
        # Crear modelo
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar con pérdidas y métricas para cada salida
        losses = {f'{category}_output': 'sparse_categorical_crossentropy' for category in num_classes_dict.keys()}
        metrics = {f'{category}_output': 'accuracy' for category in num_classes_dict.keys()}
        
        model.compile(
            optimizer='adam',
            loss=losses,
            metrics=metrics
        )
        
        return model
    
    def train_model(self, X, y_encoded, epochs=50, batch_size=32):
        """Entrena el modelo multi-output"""
        print("\nEntrenando modelo multi-output...")
        
        # Preparar datos de salida
        y_train_dict = {}
        y_val_dict = {}
        
        # Dividir datos
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_train, X_val = X[train_idx], X[val_idx]
        
        for category in y_encoded.keys():
            y_train_dict[f'{category}_output'] = y_encoded[category][train_idx]
            y_val_dict[f'{category}_output'] = y_encoded[category][val_idx]
        
        # Crear modelo
        num_classes_dict = {category: len(np.unique(y_encoded[category])) for category in y_encoded.keys()}
        self.model = self.create_multi_output_model(num_classes_dict)
        
        print(f"Arquitectura del modelo:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Augmentación de datos
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Entrenar
        history = self.model.fit(
            datagen.flow(X_train, y_train_dict, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val_dict),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar
        val_results = self.model.evaluate(X_val, y_val_dict, verbose=0)
        print(f"\nResultados de validación:")
        for i, category in enumerate(y_encoded.keys()):
            acc_idx = i * 2 + 1  # Las métricas de accuracy están en posiciones impares
            if acc_idx < len(val_results):
                print(f"{category}: {val_results[acc_idx]:.4f}")
        
        return history
    
    def convert_to_tflite(self):
        """Convierte el modelo a formato TensorFlow Lite"""
        print("Convirtiendo modelo a TFLite...")
        
        # Crear el convertidor
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Configurar optimizaciones
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convertir
        tflite_model = converter.convert()
        
        # Guardar
        tflite_filename = 'multi_classifier.tflite'
        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Modelo TFLite guardado como: {tflite_filename}")
        return tflite_filename
    
    def save_label_encoders(self):
        """Guarda los encoders de etiquetas"""
        encoders_data = {}
        for category, encoder in self.label_encoders.items():
            encoders_data[category] = {
                'classes': encoder.classes_.tolist(),
                'class_to_index': {cls: idx for idx, cls in enumerate(encoder.classes_)}
            }
        
        with open('label_encoders.json', 'w') as f:
            json.dump(encoders_data, f, indent=2)
        
        print("Encoders de etiquetas guardados en: label_encoders.json")
    
    def plot_training_history(self, history):
        """Grafica el historial de entrenamiento"""
        categories = list(self.label_encoders.keys())
        fig, axes = plt.subplots(2, len(categories), figsize=(15, 8))
        
        for i, category in enumerate(categories):
            # Precisión
            acc_key = f'{category}_output_accuracy'
            val_acc_key = f'val_{category}_output_accuracy'
            
            if acc_key in history.history:
                axes[0, i].plot(history.history[acc_key], label='Entrenamiento')
                axes[0, i].plot(history.history[val_acc_key], label='Validación')
                axes[0, i].set_title(f'Precisión - {category}')
                axes[0, i].set_xlabel('Época')
                axes[0, i].set_ylabel('Precisión')
                axes[0, i].legend()
            
            # Pérdida
            loss_key = f'{category}_output_loss'
            val_loss_key = f'val_{category}_output_loss'
            
            if loss_key in history.history:
                axes[1, i].plot(history.history[loss_key], label='Entrenamiento')
                axes[1, i].plot(history.history[val_loss_key], label='Validación')
                axes[1, i].set_title(f'Pérdida - {category}')
                axes[1, i].set_xlabel('Época')
                axes[1, i].set_ylabel('Pérdida')
                axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()


class ImagePredictor:
    """Clase para usar el modelo entrenado para hacer predicciones"""
    
    def __init__(self):
        self.interpreter = None
        self.label_encoders = {}
        self.img_size = (256, 256)
        self.input_details = None
        self.output_details = None
    
    def load_model_and_encoders(self):
        """Carga el modelo TFLite y los encoders"""
        # Cargar encoders
        try:
            with open('label_encoders.json', 'r') as f:
                self.label_encoders = json.load(f)
        except FileNotFoundError:
            print("Error: No se encontró label_encoders.json. Ejecuta primero el entrenamiento.")
            return False
        
        # Cargar modelo TFLite
        try:
            self.interpreter = tf.lite.Interpreter(model_path='multi_classifier.tflite')
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("Modelo multi-output cargado exitosamente")
            print(f"Entradas: {len(self.input_details)}")
            print(f"Salidas: {len(self.output_details)}")
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
        
        return True
    
    def preprocess_image(self, image_path):
        """Preprocesa una imagen para la predicción"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Agregar dimensión de batch
        return img
    
    def predict(self, image_path):
        """Hace predicciones para todas las categorías"""
        if not self.interpreter:
            if not self.load_model_and_encoders():
                return None
        
        img = self.preprocess_image(image_path)
        
        # Hacer predicción
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        
        # Obtener resultados
        results = {}
        categories = ['body_type', 'gender', 'skin_color']
        
        for i, category in enumerate(categories):
            output_data = self.interpreter.get_tensor(self.output_details[i]['index'])
            
            # Decodificar resultado
            predicted_idx = np.argmax(output_data[0])
            confidence = output_data[0][predicted_idx]
            predicted_class = self.label_encoders[category]['classes'][predicted_idx]
            
            results[category] = {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': {
                    cls: float(output_data[0][j]) 
                    for j, cls in enumerate(self.label_encoders[category]['classes'])
                }
            }
        
        return results
    
    def predict_batch(self, image_paths):
        """Hace predicciones para múltiples imágenes"""
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                if result:
                    result['image_path'] = img_path
                    results.append(result)
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
        return results


def train_models():
    """Función para entrenar el modelo multi-output"""
    # Verificar que existe la carpeta dataset
    if not os.path.exists('dataset'):
        print("Error: No se encontró la carpeta 'dataset' en el directorio actual.")
        print("Asegúrate de que la estructura sea:")
        print("dataset/")
        print("  body_type/")
        print("    average/")
        print("    chubby/")
        print("    hourglass/")
        print("    skinny/")
        print("  gender/")
        print("    female/")
        print("    male/")
        print("  skin_color/")
        print("    black/")
        print("    brown/")
        print("    light/")
        return
    
    # Crear el entrenador
    trainer = MultiOutputImageClassifier()
    
    # Cargar y preprocesar datos
    X, y_encoded = trainer.load_and_preprocess_data()
    
    if len(X) == 0:
        print("Error: No se encontraron imágenes en el dataset.")
        return
    
    print(f"\n{'='*50}")
    print("ENTRENANDO MODELO MULTI-OUTPUT")
    print(f"{'='*50}")
    
    # Entrenar modelo
    history = trainer.train_model(X, y_encoded)
    
    # Convertir a TFLite
    tflite_filename = trainer.convert_to_tflite()
    
    # Guardar modelo de Keras
    trainer.model.save('multi_classifier.h5')
    print("Modelo Keras guardado como: multi_classifier.h5")
    
    # Guardar encoders de etiquetas
    trainer.save_label_encoders()
    
    # Graficar historial de entrenamiento
    trainer.plot_training_history(history)
    
    print(f"\n{'='*50}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*50}")
    print("Archivos generados:")
    print(f"  - {tflite_filename}")
    print("  - multi_classifier.h5")
    print("  - label_encoders.json")
    print("  - training_history.png")


def test_prediction(image_path):
    """Función de ejemplo para probar una predicción"""
    predictor = ImagePredictor()
    
    if not predictor.load_model_and_encoders():
        print("No se pudo cargar el modelo. ¿Ya entrenaste el modelo?")
        return
    
    try:
        results = predictor.predict(image_path)
        print(f"\nResultados para: {image_path}")
        print("-" * 50)
        for category, pred in results.items():
            print(f"{category:12}: {pred['prediction']:12} (confianza: {pred['confidence']:.2f})")
            
        print(f"\nProbabilidades detalladas:")
        print("-" * 50)
        for category, pred in results.items():
            print(f"\n{category}:")
            for cls, prob in pred['all_probabilities'].items():
                print(f"  {cls:12}: {prob:.3f}")
            
        return results
    except Exception as e:
        print(f"Error haciendo predicción: {e}")


def test_multiple_images(image_folder):
    """Prueba el modelo con múltiples imágenes de una carpeta"""
    predictor = ImagePredictor()
    
    if not predictor.load_model_and_encoders():
        print("No se pudo cargar el modelo.")
        return
    
    # Buscar imágenes en la carpeta
    image_paths = []
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_paths.append(os.path.join(image_folder, file))
    
    if not image_paths:
        print(f"No se encontraron imágenes en {image_folder}")
        return
    
    print(f"Probando {len(image_paths)} imágenes...")
    results = predictor.predict_batch(image_paths)
    
    # Mostrar resultados resumidos
    for result in results:
        img_name = os.path.basename(result['image_path'])
        print(f"{img_name:20} -> ", end="")
        for category in ['body_type', 'gender', 'skin_color']:
            pred = result[category]['prediction']
            conf = result[category]['confidence']
            print(f"{category}: {pred}({conf:.2f}) ", end="")
        print()


def show_usage():
    """Muestra cómo usar el script"""
    print("\n" + "="*60)
    print("CLASIFICADOR MULTI-OUTPUT DE IMÁGENES")
    print("="*60)
    print("VENTAJAS DEL MODELO ÚNICO:")
    print("Un solo archivo .tflite (más pequeño)")
    print("Predicción más rápida (una sola pasada)")
    print("Características compartidas entre categorías")
    print("Menos memoria RAM requerida")
    print()
    print("CÓMO USAR:")
    print("1. ENTRENAR:")
    print("   train_models()")
    print()
    print("2. PROBAR UNA IMAGEN:")
    print("   test_prediction('ruta/imagen.jpg')")
    print()
    print("3. PROBAR MÚLTIPLES IMÁGENES:")
    print("   test_multiple_images('carpeta_con_imagenes')")
    print()
    print("4. USAR DIRECTAMENTE:")
    print("   predictor = ImagePredictor()")
    print("   predictor.load_model_and_encoders()")
    print("   results = predictor.predict('imagen.jpg')")
    print("="*60)


if __name__ == "__main__":
    # Configurar GPU si está disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU disponible: {len(gpus)} dispositivos")
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("Usando CPU para el entrenamiento")
    
    show_usage()
    
    # Descomenta la línea siguiente para entrenar automáticamente
    # train_models()
    
    # Ejemplo de uso después del entrenamiento:
    # test_prediction('ruta/a/tu/imagen.jpg')
    # test_multiple_images('carpeta_test')
