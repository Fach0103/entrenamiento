Este proyecto de inteligencia artificial tiene como objetivo clasificar imágenes de residuos en cinco categorías principales: plástico, papel, vidrio, orgánico y metal. Utiliza visión computacional y aprendizaje profundo mediante la arquitectura MobileNetV2, entrenada con TensorFlow y desplegada a través de una interfaz interactiva construida con Streamlit. El sistema permite entrenar el modelo, evaluarlo con métricas de precisión y matriz de confusión, y probarlo en tiempo real subiendo imágenes desde una interfaz web.

La estructura del proyecto debe estar organizada dentro de la carpeta ia/entrenamiento, que contiene tres archivos principales: entrenamiento_modelo.py para entrenar el modelo, evaluacion_modelo.py para evaluar su rendimiento, y residuos_app.py para ejecutar la interfaz de clasificación. Además, debe existir una carpeta llamada dataset/ dentro de entrenamiento, que contiene subcarpetas por clase, como plastico/, papel/, vidrio/, organico/ y metal/, cada una con imágenes en formato .jpg, .jpeg o .png.

Para que el modelo funcione correctamente, se recomienda tener al menos 30 imágenes por clase, aunque el sistema puede entrenar con menos para pruebas iniciales. Las imágenes deben estar bien etiquetadas y distribuidas en sus respectivas carpetas. Puedes obtener datasets confiables desde TrashNet (GitHub), TACO (Trash Annotations in Context) o Kaggle (Waste Categories Dataset), que ofrecen imágenes clasificadas por tipo de residuo.

Antes de ejecutar el proyecto, es necesario instalar las dependencias con el siguiente comando en la terminal:

pip install numpy tensorflow matplotlib seaborn scikit-learn pillow streamlit

Una vez instalado todo, abre la terminal en Visual Studio Code y navega a la carpeta del proyecto con:

cd C:\Users\pc\Desktop\ia\entrenamiento

Desde allí, puedes entrenar el modelo con:

python entrenamiento_modelo.py

Esto cargará las imágenes, aplicará aumentación de datos, entrenará el modelo y guardará el archivo modelo_entrenado.h5. Luego puedes evaluar el modelo con:

python evaluacion_modelo.py

Este archivo mostrará métricas de precisión y una matriz de confusión para validar el rendimiento del modelo. Finalmente, puedes ejecutar la interfaz con:

streamlit run residuos_app.py

Esto abrirá una página web donde puedes subir una imagen de residuo y ver la clasificación automática con el porcentaje de confianza.

Si en algún momento aparece un error como "PyDataset has length 0", significa que el sistema no encontró imágenes en el dataset. En ese caso, verifica que las subcarpetas contengan imágenes válidas y que no estén vacías. Puedes usar comandos en PowerShell como:

Get-ChildItem -Recurse -Include *.jpg, *.jpeg, *.png .\dataset\ | Measure-Object

para contar imágenes, o:

Get-ChildItem .\dataset\* -Recurse -Include *.jpg, *.jpeg, *.png | Group-Object Directory | Select-Object Name, Count
