

Svara gärna kort och koncist.
1. All data i Python representeras av objekt. I Python består varje objekt av tre
grundläggande delar. Vilka är dessa delar? Beskriv varje del kort.

Identitet - Objektets unika adress

Värde - Faktiskt data som objekt innehåller. Int/tal 88 eller sträng "hej"

Typ - vilken klass objektet tillhör som bestämmer vad man kan göra med det.


2. Förklara skillnaden mellan mutabla och immutabla objekt i Python och ge
exempel på varje.

Immutabla - Värdet kan inte ändras efter att det skapats, exempelen är int, str, tuple, float

x = "kebab"
x += "sås"


Mutabla - Värdet kan ändras på plats, exemplen är list, dict, set.

lst = [1, 2]
lst.append(3)

3. Vad är ett set i Python? Vilka egenskaper särskiljer det från listor? Ge ett praktiskt
exempel där det är mer effektivt att använda ett set än en lista.

Ett set struntar i ordning och tillåter bara unika värden medans en lista är ordnad och tillåter dubbletter.

Skillnader mot listor, inga dubletter och oordnat

Exemplen kan t ex vara mejl-adresser eller unika besökare på en hemsida.




4. Förklara vad en loop är och ge exempel på när den används.
Ett sätt att skriva kod så att den körs igen istället för att man ska skriva om samma kod igen. While loop är ett exempel(for-loop finns också), om du ska logga in på en hemsida så får använderen försöka tills de skriver rätt lösenord.

5. Vad är en klass och hur skiljer sig en instans från själva klassen?

En klass är som en ritning, en instans är t ex en faktisk bil byggd från den ritningen. Klassen beskriver vad en bil har (färg, hastighet), instansen är en specifik bil med faktiska värden.


6. Vad är en funktion och varför använder programmerare dem? Förklara skillnaden
mellan en funktion med returvärde och en som inte returnerar något.

Du har ett kodblock som du kan återanvända istället för att skriva samma funktion överallt. Det blir också enklare att underhålla, om du ska göra en ändring i funktionen så görs det på ett ställe, inte på flera.

Returvärde kan vara som en miniräknare, du skickar värden och får svar tillbaka.

Utan returvärde blir att den gör något eller säger något.

7. Förklara begreppet parameter jämfört med argument Python.
En parameter är variabelnamnet i funktionens definition. Ett argument är det faktiska värdet du skickar in när du anropar funktionen.

parameter = namnet på lådan
argument = det du lägger i lådan


8. Titta på följande video om R: R Programming. Vad är skillnaden mellan R och
Python?

R använder vektorer där du kan ha flera värden av samma typ.
Python använder en lista men kan köra olika typ.

Exempel på Data Structures i R: 

Vectors
Arrays
Matrices
Factors

Detta finns inbygt. I python behöver du importera bibliotek. 

Vectors -> list
Arrays -> numpy.array 


Andra styrkor i R
Statistiska funktioner 
Linjär regression

Tidyverse är ett helt ekosystem inom R där funktioner och syntax är mer läsbara och konsekventa för dataanalys vilket Python inte har.

I det ingår bland annat

Tibble
dplyr
ggplot2
tidyr

9. Kalle ska bygga en ML-modell och delar upp sin data i ”Träning”, ”Validering” och
”Test”, vad används respektive del för?

Träning – Data modellen lär sig på, som att plugga inför ett prov

Validering – Används under träningen för att finjustera och kolla att modellen inte overfittar. 

Test – Helt ny data modellen aldrig sett. Det riktiga provet. Visar hur bra modellen faktiskt presterar på okänd data.



10. Julia delar upp sin data i träning och test. På träningsdatan så tränar hon tre
modeller; ”Linjär Regression”, ”Lasso regression” och en ”Random Forest
modell”. Hur skall hon välja vilken av de tre modellerna hon skall fortsätta
använda när hon inte skapat ett explicit ”valideringsdataset”?

Hon kan använda sig av korsvalidering. Hon delar upp träningsdatan i t ex 5 delar, tränar på 4 och testar på 5e, roterar oh jämför snittresultat. Den model med bäst snittpoäng vinner, sen testar hon den  på testdatan för slutgiltigt resultat.


11. Vad är ”regressionsproblem? Kan du ge några exempel på modeller som används
och potentiella tillämpningsområden?

När du försöker förutsäga ett numeriskt värde. Exempel är förutsäga huspriser, temperatur eller aktiekurser. Modeller kan vara Linjär Regression och Random Forest.

12. Hur kan du tolka RMSE och vad används det till.
 Mäter hur mycket modellens förutsägelser avviker från de verkliga värdena. Så ju lägre RMSE du har desto bättre modell har du.



13. Vad är ”klassificieringsproblem? Kan du ge några exempel på modeller som
används och potentiella tillämpningsområden?

Klassificeringsproblem är när du förutsäger en kategori istället för tal.
Modeller kan vara Logistisk Regression eller Random Forest.

14. Vad är en ”Confusion Matrix”?

Det är en tabell som visar hur bra en klassificeringsmodell presterar.
Den visar hur många gånger den gissade rätt med True positive/negative
och hur många den missade med false positive/negative

15. Vad är K-means modellen för något? Ge ett exempel på vad det kan tillämpas på.

K-means grupperar data i antal kluster baserat på likhet. Exempel är att gruppera kunder baserat på köpbeteende - Storkunder, budgetkunder, VIPkunder.


16. Förklara (gärna med ett exempel): Ordinal encoding, one-hot encoding, dummy
variable encoding.


Ordinal encoding - Kategorier med naturlig rangordning får siffror där ordningen spelar roll.
Grundskola = 1
Gymnasium = 2 
Universitet = 3 

One-hot encoding – Varje kategori blir en egen kolumn med 0 eller 1, ingen rangordning
Röd = [1.0.0]
Blå = [0.1.0]
Grön = [0.0.1]

Dummy variable encoding – Samma som one-hot men du droppar en kolumn. Om Röd=0 och Blå=0 vet du att det är Grön. Slipper redundans.


17. Göran påstår att datan antingen är ”ordinal” eller ”nominal”. Julia säger att detta måste tolkas. Hon ger ett exempel med att färger såsom {röd, grön, blå} generellt sett inte har någon inbördes ordning (nominal) men om du har en röd skjorta så är du vackrast på festen (ordinal) – vem har rätt?

Julia - Data kan tolkas olika beroende på sammanhang. Det handlar om hur du väljer att använda datan.


18. Vad är skillnaden mellan parametrar och hyperparametrar i en
maskininlärningsmodell? Ge ett exempel på varje och förklara varför de inte kan
optimeras på samma sätt.


 I parametrar lär sig modellen själv under träning. Hyperparametrar ställer du in manuellt innan träning. Parametrar optimeras av algoritmen, hyperparametrar måste du testa dig fram till – t.ex. med korsvalidering.


19. Förklara skillnaden mellan overfitting och underfitting i en
maskininlärningsmodell. Beskriv även hur man kan upptäcka respektive åtgärda
dem

Overfitting – modellen har "memorerat" träningsdatan istället för att lära sig mönster. Presterar bra på träning, dåligt på test. mer data, enklare modell, regularisering.


Underfitting – modellen är för enkel och missar mönster. Presterar dåligt på både träning och test. Åtgärd: mer komplex modell, fler features, träna längre.



-----------------------

Du ska även genomföra en självutvärdering där du besvarar följande tre frågor

1. Har något varit utmanande i kursen/kunskapskontrollerna? Om ja, hur har du
hanterat det?

För mig personligen kan det ha blivit lite för mycket ibland på lektionerna, det är inte att lektionerna har varit dåliga men det kanske har varit lite för långa förklaringar vilket i sin tur lett till att man står där och ba jahop, nu fattar jag ju ingenting? 

Detta löste sig genom att jag faktiskt testade saker för att enklare förstå. För mig är det oftast mycket enklare och bättre. Att t ex sitta och fixa i c++ har ju enbart fungerat för att jag vågat testa mig fram, samma sak gäller för python. 

2. Vilket betyg anser du att du ska ha och varför?
Jag hoppas att jag gjort mig förtjänt av ett VG, då jag bör ha uppfyllt det kursen kräver.

3. Något du vill lyfta fram till Terese?

Nej då allt är super bra! Stort tack! :-)
