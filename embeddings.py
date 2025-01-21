from sklearn.decomposition import PCA
from dotenv import load_dotenv
import numpy as np
import requests
import os

load_dotenv()

class Embeddings:
    def __init__(self):
        self.api_key = os.getenv("hf_token")
        self.api_model = os.getenv("model_id")
        self.pca = PCA(n_components=3)
        
        if not self.api_key or not self.api_model:
            raise EnvironmentError("Las variables de entorno 'hf_token' y/o 'model_id' no están configuradas.")

        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.api_model}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_embeddings(self, text):
        if not text:
            raise ValueError("El texto de entrada no puede estar vacío.")

        try:
            response = requests.post(self.api_url, headers=self.headers, json={"inputs": text, "options": {"wait_for_model": True}})
            response.raise_for_status()
           
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_embedding_and_chunks(self, pdf_text, texto, chunk_size, chunk_overlap):
        """
        Args:
            pdf_text (str): Texto extraído de un PDF.
        """
        chunks = self.get_chunks(pdf_text, chunk_size, chunk_overlap)
        embeddings = [self.get_embeddings(chunk) for chunk in chunks]
        embeddings.append(self.get_embeddings(texto))
        array = np.array(embeddings)

        pca_result = self.pca.fit_transform(array)
        return pca_result, chunks
    
    def get_chunks(self, pdf_text, chunk_size, overlap):
        """
        Divide el texto extraído de un PDF en fragmentos más pequeños llamados chunks.

        Args:
            pdf_text (str): Texto extraído de un PDF.
            chunk_size (int): Tamaño de los fragmentos de texto.
            overlap (int): Número de caracteres que se superponen entre fragmentos.
        
        Returns:
            Iterator: Generador de chunks.
        """
        chunks = []
        start = 0
        while start < len(pdf_text):
            end = min(start + chunk_size, len(pdf_text))
            chunk = pdf_text[start:end]

            if len(chunks) >= 1:
                chunk = chunks[-1][-overlap:] + chunk
            
            chunks.append(chunk)
            start = end
        return chunks
        

if __name__ == "__main__":
    try:
        embeddings = Embeddings()
        emb, chunks = embeddings.get_embedding_and_chunks("""
NOMBRE: Daniel Naranjo García
 EI1038/EI1041
 Práctica 7. Vistas y disparadores en PostgreSQL y Oracle
 Objetivos
 ● Sercapazdedeterminar los roles de los usuarios de la aplicación.
 ● Ser capaz de diseñar e implementar vistas que permitan garantizar la seguridad de la
 aplicación.
 ● Sercapazdeactualizar datos a través de las vistas.
 Rellena los diferentes apartados en color azul.
 1. Entrega y evaluación
 Para que la realización de esta práctica forme parte de tu evaluación deberás realizarla en el aula de
 prácticas y hacer la entrega al finalizar la sesión (en papel o a través del aula virtual). Si no puedes
 asistir a la clase correspondiente, puedes realizar la práctica y entregarla antes de la sesión de
 laboratorio en que se hará la primera sesión de la práctica, es decir, antes de las 15:00 del viernes.
 La práctica la vamos a realizar en postgreSQL y en ORACLE.
 2. La base de datos
 La práctica se basa en la base de datos de tu proyecto. Concretamente vamos a trabajar sobre las
 tablas TRABAJADOR, ESPECIALISTA, REPARA y SENSOR, y en las relaciones entre ellas. Como
 verás, hay alguna modificación con respecto al proyecto. Además del nombre de las tablas (que
 puede que en tu modelo no se llamen igual), hemos eliminado la entidad técnico para que la vista sea
 más fácil.
 El diseño físico en el que se va a basar la práctica lo tienes en un fichero enlazado junto al boletín.
 Antes de nada, estudia el diseño propuesto. También tienes un fichero con los datos a insertar en la
 base de datos para la práctica.
 3. Primero probamos con PostgreSQL
 1. Crea una base de datos con el nombre de tu usuario alxxxxx_P7. Una vez hayas creado la base
 de datos, conéctate a ella y ejecuta las sentencias del fichero de creación de tablas. Recuerda borrar
 la base de datos cuando acabes.
Curso 2024/25
 EI1038/41
 NOMBRE: Daniel Naranjo García
 3.1 Vistas
 2. Escribe una vista estado_sensores_exp que muestre los sensores que reparan los especialistas
 que tienen menos de un año de experiencia. Define la vista para que su resultado sea similar al
 siguiente:
 create view estado_sensores_exp as
 select t.dni, t.nombre, t.apellidos, t.mail, s.cod_sensor, s.estado, r.fecha_repara
 from trabajador as t join especialista as a using(dni) join repara as r using(dni)
 join sensor as s using(cod_sensor);
 3.2 Actualizar los datos a través de la vista
 Cuando un administrativo/a está usando la vista, se da cuenta de que hay especialistas que no han
 actualizado bien la fecha de reparación del sensor y deben actualizarla.
 3. ¿Crees que se puede actualizar la vista directamente? ¿Por qué?
 No se va a poder porque la vista contiene información de diferenes tables por lo que al intentar hacer
 el update, saltará un error
Curso 2024/25
 EI1038/41
 NOMBRE: Daniel Naranjo García
 4. Escribe una sentencia de actualización de las fecha de reparación del especialista con dni
 '00000003C' para el sensor 3 y comprueba si el resultado es el esperado.
 update estado_sensores_exp set fecha_repara='01-01-2025' WHERE dni='00000003C' AND
 cod_sensor=3;
 5. Si no puedes actualizar la vista, crea un disparador que permita hacerlo.
 CREATE ORREPLACETRIGGERtrg_ej5
 INSTEAD OF UPDATE on estado_sensores_exp
 FOREACHROW
 EXECUTE PROCEDUREfunEj5();
 CREATE ORREPLACEFUNCTIONfunEj5()
 RETURNS TRIGGER AS$$
 BEGIN
 END;
 UPDATE repara
 SET fecha_repara=NEW.fecha_repara
 WHEREdni=NEW.dni AND cod_sensor=NEW.cod_sensor;
 RETURNNEW;
 $$ language 'plpgsql';
 4. Ahora probamos con ORACLE
 Conéctate a Oracle y crea las mismas tablas. Para ello ejecuta las sentencias del fichero de creación
 de tablas. También tienes que insertar los datos.
 4.1 Vistas
 6. Escribe la misma vista estado_sensores_exp. Recuerda que debe mostrar los sensores que
 reparan los especialistas que tienen menos de un año de experiencia.
 CREATE ORREPLACEVIEWestado_sensores_exp AS
 SELECT
Curso 2024/25
 EI1038/41
 NOMBRE: Daniel Naranjo García
 t.dni,
 t.nombre,
 t.apellidos,
 t.mail,
 s.cod_sensor,
 s.estado,
 r.fecha_repara
 FROM
 trabajador t
 INNER JOIN especialista a ON t.dni = a.dni
 INNER JOIN repara r ON t.dni = r.dni
 INNER JOIN sensor s ON r.cod_sensor = s.cod_sensor;
 4.2 Actualizar los datos a través de la vista
 Cuando un administrativo/a está usando la vista, se da cuenta de que hay especialistas que no han
 actualizado bien la fecha de reparación del sensor y deben actualizarla.
 7. ¿Crees que se puede actualizar la vista directamente? ¿Por qué?
 Al intentar actualizar la vista pasaría lo mismo que con Postgres y lanzará un mensaje de error
 debido a que la vista está compuesta por datos de distintas tablas.
 8. Escribe una sentencia de actualización de la fecha de reparación del especialista con dni
 '00000003C' para el sensor 2 y comprueba si el resultado es el esperado.
 La verdad que pensaba que no lo iba a actualizar, pero al ejecutar la sentencia sí que se ha
 actualizado la vista y también la tabla repara. Supongo que la actualización al afectar a una sola tabla
 Oracle permitirá eso.
 9. Si no puedes actualizar la vista, crea un disparador que permita hacerlo.
 Si que me ha dejado actualizar por lo que no voy a hacer el trigger
 10. Analiza qué ha pasado con cada SGBD y las diferencias entre ellos.
 Supongo que según el SGBD, manejará o no la actualización de datos al actualizar una vista.
 Con Postgres no ha dejado hacer la actualización mientras que con Oracle si que ha dejado debido a
 que la actualización afectaba solo a una tabla.
                                                      
""", "prueba")
        for c in chunks:
            print(c)
            print()
            print()
    except ValueError as e:
        print(f"Error: {e}")
