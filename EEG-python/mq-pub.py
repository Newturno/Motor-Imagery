# python 3.6

import random
import time

from paho.mqtt import client as mqtt_client


BROKER = '192.168.50.61'
PORT = 1883
TOPIC = "hci"
# Generate a Client ID with the publish prefix.
CLIENT_ID = f'publish-{random.randint(0, 1000)}'
USERNAME = 'dobot'
PASSWORD = '12345678#'
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(CLIENT_ID)
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client


def publish(client):
    msg_count = 1
    while True:
        time.sleep(1)
        msg = f"messages: {msg_count}"
        # main function to send
        result = client.publish(TOPIC, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{TOPIC}`")
        else:
            print(f"Failed to send message to topic {TOPIC}")
        msg_count += 1
        if msg_count > 5:
            break


def run():
    client = connect_mqtt()
    #client.loop_start()
    publish(client)
    #client.loop_stop()


if __name__ == '__main__':
    run()
