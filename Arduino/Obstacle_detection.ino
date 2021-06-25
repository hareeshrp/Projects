#include"Ultrasonic.h"
Ultrasonic ultrasonic(6,7);
const int light = 13;
long microsec = 0;
float distance = 0;

void setup
{
    Serial.begin(9600);
    pinMode(light,OUTPUT);
}

void loop()
{
    microsec = ultrasonic.timing();
    distance = ultrasonic.convert(microsec,Ultrasonic::CM);
    led ultra();
    Serial.print(distance);
    Serial.println("cm");
    delay(1000);
}

void ultra()
{
    digitalWrite(light,LOW);
    if (distance > 5)
    {
        digitalWrite(light,HIGH);
    }
}

