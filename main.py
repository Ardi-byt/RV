import cv2 as cv
import numpy as np

def zmanjsaj_sliko(slika, sirina, visina):
    '''Zmanjšaj sliko na velikost sirina x visina.'''
    return cv.resize(slika, (sirina, visina))

def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze) -> list:
    '''Sprehodi se skozi sliko v velikosti škatle (sirina_skatle x visina_skatle) in izračunaj število pikslov kože v vsaki škatli.
    Škatle se ne smejo prekrivati!
    Vrne seznam škatel, s številom pikslov kože.
    Primer: Če je v sliki 25 škatel, kjer je v vsaki vrstici 5 škatel, naj bo seznam oblike
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]]. 
      V tem primeru je v prvi škatli 1 piksel kože, v drugi 0, v tretji 0, v četrti 1 in v peti 1.'''

    spodnja_meja, zgornja_meja = barva_koze
    skatle = []
    for y in range(0, slika.shape[0] - visina_skatle + 1, visina_skatle):
        for x in range(0, slika.shape[1] - sirina_skatle + 1, sirina_skatle):
            okno = slika[y:y + visina_skatle, x:x + sirina_skatle]
            piksli_koze = prestej_piklse_z_barvo_koze(okno, (spodnja_meja, zgornja_meja))
            skatle.append((x, y, piksli_koze))
    return skatle

def prestej_piklse_z_barvo_koze(slika, barva_koze) -> int:
    '''Prestej število pikslov z barvo kože v škatli.'''
    spodnja_meja, zgornja_meja = barva_koze
    maska = cv.inRange(slika, spodnja_meja, zgornja_meja)
    return cv.countNonZero(maska)

def doloci_barvo_koze(slika, levo_zgoraj, desno_spodaj) -> tuple:
    '''Ta funkcija se kliče zgolj 1x na prvi sliki iz kamere. 
    Vrne barvo kože v območju ki ga definira oklepajoča škatla (levo_zgoraj, desno_spodaj).
      Način izračuna je prepuščen vaši domišljiji.'''
    x1, y1 = levo_zgoraj
    x2, y2 = desno_spodaj
    izbrano_obmocje = slika[y1:y2, x1:x2]
    
    povprecje = np.mean(izbrano_obmocje, axis=(0, 1))
    std_dev = np.std(izbrano_obmocje, axis=(0, 1))
    
    spodnja_meja = np.maximum(povprecje - std_dev, 0).astype(np.uint8)
    zgornja_meja = np.minimum(povprecje + std_dev, 255).astype(np.uint8)
    
    spodnja_meja_1_3 = np.array([[spodnja_meja[0], spodnja_meja[1], spodnja_meja[2]]], dtype=np.uint8)
    zgornja_meja_1_3 = np.array([[zgornja_meja[0], zgornja_meja[1], zgornja_meja[2]]], dtype=np.uint8)
    
    return (spodnja_meja_1_3, zgornja_meja_1_3)

if __name__ == '__main__':
    # Pripravi kamero
    camera = cv.VideoCapture(0)

    # Preveri, ali je kamera odprta
    if not camera.isOpened():
        print("Ni mogoče odpreti kamere")
        exit()

    # Zajemi prvo sliko iz kamere
    success, frame = camera.read()
    if not success:
        print("Ni mogoče zajeti slike iz kamere")
        exit()

    zmanjsana_slika = zmanjsaj_sliko(frame, 240, 320)

    #Izberes obmocje obraza
    r = cv.selectROI("Izberi območje obraza", zmanjsana_slika, fromCenter=False, showCrosshair=True)
    cv.destroyWindow("Izberi območje obraza")
    
    x, y, w, h = map(int, r)
    levo_zgoraj = (x, y)
    desno_spodaj = (x + w, y + h)
    
    barva_koze = doloci_barvo_koze(zmanjsana_slika, levo_zgoraj, desno_spodaj)
    print(f"Določena barva kože: spodnja meja = {barva_koze[0]}, zgornja meja = {barva_koze[1]}")

    sirina_skatle = int(240 * 0.1)
    visina_skatle = int(320 * 0.1)

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        zmanjsana_slika = zmanjsaj_sliko(frame, 240, 320)

        skatle = obdelaj_sliko_s_skatlami(zmanjsana_slika, sirina_skatle, visina_skatle, barva_koze)
        
        # Narisi skatle na sliki
        for x, y, piksli_koze in skatle:
            if piksli_koze > 100:
                cv.rectangle(zmanjsana_slika, (x, y), (x + sirina_skatle, y + visina_skatle), (0, 255, 0), 2)

        izbrano_obmocje = zmanjsana_slika[y:y+h, x:x+w]
        st_pikslov_koze = prestej_piklse_z_barvo_koze(izbrano_obmocje, barva_koze)
    
        cv.imshow('Live kamera', zmanjsana_slika)
    
        # Pocaka da uporabnik stisne q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Osvoboditev kamere
    camera.release()
    cv.destroyAllWindows()
