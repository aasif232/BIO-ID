#include <WiFi.h>
#include <Adafruit_Fingerprint.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// OLED Setup (0.91-inch I2C SSD1306)
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET    -1
#define SCREEN_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

const char* ssid = "Fawstech R&D";
const char* password = "R&D@Fawstech";

WiFiServer wifiServer(8080);
HardwareSerial mySerial(2); // RX=16, TX=17
Adafruit_Fingerprint finger = Adafruit_Fingerprint(&mySerial);

void updateDisplay(String line1, String line2 = "") {
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(line1);
  display.setCursor(0, 18); // Slightly lower for the second line
  display.println(line2);
  display.display();
}

void setup() {
  Serial.begin(115200);
  
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    for(;;); 
  }
  display.setTextColor(SSD1306_WHITE);
  updateDisplay("Connecting WiFi...");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  
  updateDisplay("Connected!", WiFi.localIP().toString());
  delay(2000);

  mySerial.setRxBufferSize(4096); 
  mySerial.begin(57600, SERIAL_8N1, 16, 17);
  finger.begin(57600);
  
  if (!finger.verifyPassword()) {
    updateDisplay("Sensor Error");
    while(1) delay(1);
  }

  updateDisplay("Ready", "Waiting for App...");
  wifiServer.begin();
}

void loop() {
  WiFiClient client = wifiServer.available();

  if (client) {
    updateDisplay("Client Connected", "Scan Finger...");
    
    while (finger.getImage() != FINGERPRINT_OK) { 
      if (!client.connected()) return;
      delay(10); 
    }
    
    updateDisplay("Uploading...", "Please wait");
    
    // Clear serial and send upload command
    while(mySerial.available()) mySerial.read(); 
    uint8_t upImageCmd[] = {0xEF, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x03, 0x0A, 0x00, 0x0E};
    mySerial.write(upImageCmd, sizeof(upImageCmd));

    uint32_t count = 0;
    unsigned long lastByteTime = millis();
    uint8_t tempBuffer[512];

    // --- SENDER LOOP ---
    while (client.connected() && (millis() - lastByteTime < 2000) && count < 45000) {
      int available = mySerial.available();
      if (available > 0) {
        int toRead = min(available, 512);
        mySerial.readBytes(tempBuffer, toRead);
        client.write(tempBuffer, toRead);
        count += toRead;
        lastByteTime = millis();
      }
      delayMicroseconds(100); 
    }
    
    // --- RECEIVER LOOP (Waiting for Blood Group/Gender) ---
    updateDisplay("Sent. Waiting...", "Processing...");
    
    String response = "";
    unsigned long waitStart = millis();
    
    // Wait up to 5 seconds for the server to reply back
    while (client.connected() && (millis() - waitStart < 5000)) {
      if (client.available()) {
        response = client.readStringUntil('\n'); // Safely reads the entire line at once
        response.trim(); // Removes any trailing \r
        break; 
      }
    }

    if (response.length() > 0) {
      updateDisplay("Doc Locker:", response);
    } else {
      updateDisplay("No Data", "Timeout");
    }

    delay(5000); // Show the data for 5 seconds
    updateDisplay("Ready", WiFi.localIP().toString());
    client.stop();
  }
}