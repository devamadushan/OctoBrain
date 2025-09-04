
import matplotlib.pyplot as plt


class Tracer:

    def observer_tester(self, y_true, y_pred, input, mae, mse, r2, params, sorties):
        print("icii")
        plt.figure(figsize=(10, 6))
        print(y_true)
        print(y_pred)
        plt.plot(y_true, label="Données attendues", color="green")
        plt.plot(y_pred, label="Données simulées ", color="red", linestyle="dashed")
        #plt.plot(input, color="blue", label="Entré")
        plt.legend()

        plt.title(f" MAE: {round(mae, 2)} | MSE: {round(mse, 2)} | R²: {round(r2, 2)}")
       # plt.xlabel(
        #    f"x =  {', '.join(str(param) for param in params)} ::: ::: y =  {', '.join(str(sortie) for sortie in sorties)}")

        plt.show()
        return plt

    def observer_trainer(self , history_dict ):
        plt.figure(figsize=(8, 5))

        # Courbe de perte (loss)
        plt.plot(history_dict['loss'], label='Entraînement (loss)')
        if 'val_loss' in history_dict:
            plt.plot(history_dict['val_loss'], label='Validation (val_loss)')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Courbes d’apprentissage')
        plt.legend()
        plt.grid(True)
        plt.show()
