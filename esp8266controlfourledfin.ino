#include <SoftwareSerial.h>
#include <string.h>
SoftwareSerial esp8266(2,3); //Pin 2 & 3 of Arduino as RX and TX. Connect TX and RX of ESP8266 respectively.
#define DEBUG true
#define led_pin 11 //LED is connected to Pin 11 of Arduino
#define led_pin1 10 //LED is connected to Pin 10 of Arduino
#define led_pin2 9 //LED is connected to Pin 9 of Arduino
#define led_pin3 8 //LED is connected to Pin 8 of Arduino


void setup()
  {
    pinMode(led_pin, OUTPUT);
    pinMode(led_pin1, OUTPUT);
    pinMode(led_pin2, OUTPUT);
    pinMode(led_pin3, OUTPUT);

//    digitalWrite(led_pin, LOW);
    Serial.begin(9600);
    esp8266.begin(115200); //Baud rate for communicating with ESP8266. Your's might be different.
    esp8266Serial("AT+RST\r\n", 5000, DEBUG); // Reset the ESP8266
    esp8266Serial("AT+CWMODE=1\r\n", 5000, DEBUG); //Set station mode Operation
    esp8266Serial("AT+CWJAP=\"binatone\",\"coolcoolcool\"\r\n", 5000, DEBUG);//Enter your WiFi network's SSID and Password.
//     digitalWrite(led_pin, HIGH);
       delay(1000);
//       digitalWrite(led_pin, LOW);
                Serial.println("before");//Must print "led"
  /*  while(!esp8266.find("OK")) 
    {
      }*/
      Serial.println("after");
    esp8266Serial("AT+CIFSR\r\n", 5000, DEBUG);//You will get the IP Address of the ESP8266 from this command. 
    esp8266Serial("AT+CIPMUX=1\r\n", 5000, DEBUG);
    esp8266Serial("AT+CIPSERVER=1,80\r\n", 5000, DEBUG);
  }

void loop()
  {    if (esp8266.available())
      { //digitalWrite(led_pin, HIGH);
     
        if (esp8266.find("+IPD,"))
          {
            int connectionId = (esp8266.read())-48;
           String  prconn="conn=";
      Serial.println(prconn);
      Serial.println(connectionId);

            String msg;
            esp8266.find("?");
            msg = esp8266.readStringUntil('\n');
            String command1 = msg.substring(0, 3);
            String command2 = msg.substring(3,7);
                        Serial.println("before");
                String cmdCIPCLOSE = "AT+CIPCLOSE=";//Close TCP/UDP 
                cmdCIPCLOSE += connectionId;
                cmdCIPCLOSE += "\r\n";
                //changed here
                esp8266Serial(cmdCIPCLOSE, 0, DEBUG);  
                  Serial.println("after");
            if (DEBUG) 
              {
                Serial.println(command1);//Must print "led"
                Serial.println(command2);//Must print "ON" or "OFF"
              }
         //   delay(100);

                
              if (command2 == "OOOO") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, LOW);
                      digitalWrite(led_pin1, LOW);
                      digitalWrite(led_pin2, LOW);
                      digitalWrite(led_pin3, LOW);
                    }
                     else if (command2 == "OOOL") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, LOW);
                      digitalWrite(led_pin1, LOW);
                      digitalWrite(led_pin2, LOW);
                      digitalWrite(led_pin3, HIGH);
                    }else if (command2 == "OOLO") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, LOW);
                      digitalWrite(led_pin1, LOW);
                      digitalWrite(led_pin2, HIGH);
                      digitalWrite(led_pin3, LOW);
                    }else if (command2 == "OOLL") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, LOW);
                      digitalWrite(led_pin1, LOW);
                      digitalWrite(led_pin2, HIGH);
                      digitalWrite(led_pin3, HIGH);
                    }else if (command2 == "OLOO") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, LOW);
                      digitalWrite(led_pin1, HIGH);
                      digitalWrite(led_pin2, LOW);
                      digitalWrite(led_pin3, LOW);
                    }else if (command2 == "OLOL") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, LOW);
                      digitalWrite(led_pin1, HIGH);
                      digitalWrite(led_pin2, LOW);
                      digitalWrite(led_pin3, HIGH);
                    }else if (command2 == "OLLO") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, LOW);
                      digitalWrite(led_pin1, HIGH);
                      digitalWrite(led_pin2, HIGH);
                      digitalWrite(led_pin3, LOW);
                    }else if (command2 == "OLLL") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, LOW);
                      digitalWrite(led_pin1, HIGH);
                      digitalWrite(led_pin2, HIGH);
                      digitalWrite(led_pin3, HIGH);
                    }else if (command2 == "LOOO") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, HIGH);
                      digitalWrite(led_pin1, LOW);
                      digitalWrite(led_pin2, LOW);
                      digitalWrite(led_pin3, LOW);
                    }else if (command2 == "LOOL") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, HIGH);
                      digitalWrite(led_pin1, LOW);
                      digitalWrite(led_pin2, LOW);
                      digitalWrite(led_pin3, HIGH);
                    }else if (command2 == "LOLO") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, HIGH);
                      digitalWrite(led_pin1, LOW);
                      digitalWrite(led_pin2, HIGH);
                      digitalWrite(led_pin3, LOW);
                    }else if (command2 == "LOLL") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, HIGH);
                      digitalWrite(led_pin1, LOW);
                      digitalWrite(led_pin2, HIGH);
                      digitalWrite(led_pin3, HIGH);
                    }else if (command2 == "LLOO") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, HIGH);
                      digitalWrite(led_pin1, HIGH);
                      digitalWrite(led_pin2, LOW);
                      digitalWrite(led_pin3, LOW);
                    }else if (command2 == "LLOL") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, HIGH);
                      digitalWrite(led_pin1, HIGH);
                      digitalWrite(led_pin2, LOW);
                      digitalWrite(led_pin3, HIGH);
                    }else if (command2 == "LLLO") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, HIGH);
                      digitalWrite(led_pin1, HIGH);
                      digitalWrite(led_pin2, HIGH);
                      digitalWrite(led_pin3, LOW);
                    }else if (command2 == "LLLL") 
                    {
                      Serial.println("hell");
                      digitalWrite(led_pin, HIGH);
                      digitalWrite(led_pin1, HIGH);
                      digitalWrite(led_pin2, HIGH);
                      digitalWrite(led_pin3, HIGH);
                    }
                      
          }
      } 
      
  }
   
String esp8266Serial(String command, const int timeout, boolean debug)
  {
    String response = "";
    esp8266.print(command);
    long int time = millis();
    while ( (time + timeout) > millis())
      {
        while (esp8266.available())
          {
            char c = esp8266.read();
            response += c;
          }
      }
    if (debug)
      {
        Serial.print(response);
      }
    return response;
  }
 
