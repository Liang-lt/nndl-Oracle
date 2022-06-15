import numpy as np
# from pylearn.io import filetensor as ft
from PIL import Image
import pylab

class Occlusion():

    def __init__(self, seed=9854):
        # Ces 4 variables representent la taille du "crop" sur l'image2
        # Ce "crop" est pris a partie de image1[15,15], le milieu de l'image1
        self.haut = 2
        self.bas = 2
        self.gauche = 2
        self.droite = 2

        # Ces deux variables representent le deplacement en x et y par rapport
        # au milieu du bord gauche ou droit
        self.x_arrivee = 0
        self.y_arrivee = 0

        # Cette variable =1 si l'image est mise a gauche et -1 si a droite
        # et =0 si au centre, mais plus pale
        self.endroit = -1

        # Cette variable determine l'opacite de l'ajout dans le cas ou on est au milieu
        self.opacite = 0.5  # C'est completement arbitraire. Possible de le changer si voulu

        # Sert a dire si on fait quelque chose. 0=faire rien, 1 on fait quelque chose
        self.appliquer = 1

        self.seed = seed
        # np.random.seed(self.seed)
        f = Image.open('test_img/下.jpg')  # Le jeu de donnees est en local.
        w = np.asarray(f)
        # f3 = open('/data/lisa/data/ift6266h10/echantillon_occlusion.ft')  # Doit etre sur le reseau DIRO.
        # f3 = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/echantillon_occlusion.ft')
        # Il faut arranger le path sinon
        # w = ft.read(f3)
        f.close()

        self.longueur = len(w)
        # print(self.longueur)
        self.d = (w.astype('float')) / 255

    def get_settings_names(self):
        return ['haut', 'bas', 'gauche', 'droite', 'x_arrivee', 'y_arrivee', 'endroit', 'rajout', 'appliquer']

    def get_seed(self):
        return self.seed

    def regenerate_parameters(self, complexity):
        self.haut = min(13, int(np.abs(np.random.normal(int(7 * complexity), 2))))
        self.bas = min(13, int(np.abs(np.random.normal(int(7 * complexity), 2))))
        self.gauche = min(13, int(np.abs(np.random.normal(int(7 * complexity), 2))))
        self.droite = min(13, int(np.abs(np.random.normal(int(7 * complexity), 2))))
        if self.haut + self.bas + self.gauche + self.droite == 0:  # Tres improbable
            self.haut = 1
            self.bas = 1
            self.gauche = 1
            self.droite = 1

        # Ces deux valeurs seront controlees afin d'etre certain de ne pas depasser
        self.x_arrivee = int(np.abs(np.random.normal(0, 2)))  # Complexity n'entre pas en jeu, pas besoin
        self.y_arrivee = int(np.random.normal(0, 3))

        self.rajout = np.random.randint(0, self.longueur - 1)  # les bouts de quelle lettre
        self.appliquer = np.random.binomial(1, 0.4)  #####  40 % du temps, on met une occlusion #####

        if complexity == 0:  # On ne fait rien dans ce cas
            self.applique = 0

        self.endroit = np.random.randint(-1, 2)

        return self._get_current_parameters()

    def _get_current_parameters(self):
        return [self.haut, self.bas, self.gauche, self.droite, self.x_arrivee, self.y_arrivee, self.endroit,
                self.rajout, self.appliquer]

    def transform_image(self, image):
        if self.appliquer == 0:  # Si on fait rien, on retourne tout de suite l'image
            return image

        # Attrapper le bruit d'occlusion
        # print(self.rajout)
        bruit = self.d.reshape((28, 28))[13 - self.haut:13 + self.bas + 1,
                13 - self.gauche:13 + self.droite + 1]

        if self.x_arrivee + self.gauche + self.droite > 28:
            self.endroit *= -1  # On change de bord et on colle sur le cote
            self.x_arrivee = 0
        if self.y_arrivee - self.haut < -14:
            self.y_arrivee = self.haut - 14  # On colle le morceau en haut
        if self.y_arrivee + self.bas > 13:
            self.y_arrivee = 13 - self.bas  # On colle le morceau en bas

        if self.endroit == -1:  # a gauche
            for i in range(-self.haut, self.bas + 1):
                for j in range(0, self.gauche + self.droite + 1):
                    image[14 + self.y_arrivee + i, self.x_arrivee + j] = \
                        max(image[14 + self.y_arrivee + i, self.x_arrivee + j], bruit[i + self.haut, j])

        elif self.endroit == 1:  # a droite
            for i in range(-self.haut, self.bas + 1):
                for j in range(-self.gauche - self.droite, 1):
                    image[14 + self.y_arrivee + i, 27 - self.x_arrivee + j] = \
                        max(image[14 + self.y_arrivee + i, 27 - self.x_arrivee + j],
                            bruit[i + self.haut, j + self.gauche + self.droite])

        elif self.endroit == 0:  # au milieu
            for i in range(-self.haut, self.bas + 1):
                for j in range(-self.gauche, self.droite + 1):
                    image[14 + i, 14 + j] = max(image[14 + i, 14 + j],
                                                bruit[i + self.haut, j + self.gauche] * self.opacite)

        return image


# ---TESTS---

def _load_image(path):
    f = Image.open(path)  # Le jeu de donnees est en local.
    w = np.asarray(f)
    return (w / 255.0).astype('float')


def occlu_(f):
    w = np.asarray(f)
    img = (w / 255.0).astype('float')
    complexite = 0.9
    transfo = Occlusion()
    # print(img.reshape((28, 28)).shape)

        # print(img.size)
        # print
    transfo.get_settings_names()
    # print
    transfo.regenerate_parameters(complexite)

    img_trans = transfo.transform_image(img.reshape((28, 28)))

    return Image.fromarray((img_trans.reshape((28, 28)) * 255).astype('uint8'), "L")



# def occlu_():



if __name__ == '__main__':

    import scipy

    f = Image.open('test_img/下.jpg')
    occlu_(f)