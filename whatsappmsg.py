def get_time():
    import time
    t = time.localtime()
    hour = int(time.strftime("%H", t))
    min = int(time.strftime("%M", t)) +1 
    return hour, min

def send_whatsApp_msg(to, text):
    import pywhatkit
    pywhatkit.sendwhatmsg(to,text, get_time()[0], get_time()[1])

