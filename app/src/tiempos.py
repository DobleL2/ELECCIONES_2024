import datetime
import pytz

def obtener_hora():
    hora_actual = datetime.datetime.now()

    # Especificar la zona horaria deseada (en este caso, 'America/Guayaquil' para Ecuador)
    zona_horaria_ecuador = pytz.timezone('America/Guayaquil')

    # Convertir la hora actual a la zona horaria de Ecuador
    return hora_actual.astimezone(zona_horaria_ecuador).strftime('%Y-%m-%d %H:%M:%S')