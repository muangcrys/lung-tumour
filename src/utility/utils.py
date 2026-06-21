from datetime import datetime



def get_timestamp_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


