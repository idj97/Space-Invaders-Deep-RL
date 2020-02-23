**Članovi tima:**
Đorđe Ivković, SW54-2016, Grupa 4

**Asistent:**
Ivana Zeljković

**Problem koji se rešava:**
Obučavanje agenta koji maksimizuje svoj skor u igri Space Invaders Atari. Igra se igra tako što agent puca u neprijateljske vanzemaljce koji se postepeno spuštaju ka agentu, agent dobija poene za svaki uspešan pogodak a nivo se završava kada agent ubije sve vanzemaljce. igra se završava kada agent izgubi sve živote ili kada vanzemaljci stignu do dna ekrana.  Igra ima beskonačan broj nivoa gde je svaki nivo isti sa tim što protivnici postaju brži, češće gađaju agenta itd. Više o igri [ovde](https://en.wikipedia.org/wiki/Space_Invaders).
Koristiće se okruženje SpaceInvaders-v0 iz biblioteke OpenAI Gym. Stanja okruženja predstavljena su RGB slikama. Akcije agenta su mirovanje, pucanje, levo, desno, levo i pucanje, desno i pucanje. 

**Algoritam**
Zbog kompleksosti problema nije poželjno koristiti Q-learning algoritam koji bi i sa pretprocesiranjem slika zahtevao puno memorije za tabelu i sporije izračunavanje. Zbog toga će se koristiti Deep Q Learning gde se umesto tabele koristiti konvolutivna neuronska mreža (keras biblioteka) koja će za dato stanje aproksimirati Q vrednosti mogućih akcija od kojih će se birati ona najbolja. Kako bi se ubrzalo treniranje i smanjilo zauzeće memorije vršiće se pretprocesiranje slika (smanjivanje dimenzija slika i slike će se pretvarati u crno-bele). Koristiću i expirience replay tehniku. 

**Metrika za merenje performansi:**
Broj predjenih nivoa i skupljenih poena.

**Validacija rešenja:**
Posmatraće se da li se veštine agenta u odnosu na vreme treniranja povećavaju tj. da li agent nakon dužeg treniranja osvaja veći broj poena nego nakon kraćeg.
