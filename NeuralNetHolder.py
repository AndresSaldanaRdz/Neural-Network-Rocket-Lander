from Game_NN import *

class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        il, hl, ol = create_nn(2,8,2,0.3) #(il) , (hl), (ol)
        il1, hl1, ol1 = load_weights_array("weights.txt", il, hl, ol)
        self.input_l = il1
        self.hidden_l = hl1
        self.output_l = ol1

    
    def predict(self, input_row):
        
        dis_inputs = input_row.split(",")

        #Minimum Values
        X_dtt_min = -807.581052
        Y_dtt_min = 65.724787
        X_vel_min = -6.128549
        Y_vel_min = -7.820634
        #Maximum Values
        X_dtt_max = 690.524275
        Y_dtt_max = 750.667825
        X_vel_max = 8.000000
        Y_vel_max = 7.983253

        nn_input = [((float(dis_inputs[0]) - X_dtt_min) / (X_dtt_max - X_dtt_min)) , ((float(dis_inputs[1]) - Y_dtt_min) / (Y_dtt_max - Y_dtt_min))]

        il, hl, ol = forward_prop(nn_input, self.input_l, self.hidden_l, self.output_l)

        X_vel = ol[0].activation_value * (X_vel_max - X_vel_min) + X_vel_min
        Y_vel = ol[1].activation_value * (Y_vel_max - Y_vel_min) + Y_vel_min

        return (X_vel, Y_vel)