import numpy as np

class and_neural_model:

    def __init__(self, input_size, learning_rate=0.0005):
        self.weight=np.random.rand(input_size)
        self.bias=np.random.rand(1)
        print(self.weight)
        print(self.bias)
        self.learning_rate=learning_rate

    def forward_pass(self,X):
        self.N=np.dot(X,self.weight)
        self.P=self.N+self.bias
        self.Y_predict=sigmoid(self.P)
        # print(P)
        # print(Y_predict)
        return self.Y_predict
    
    def compute_loss(self,Y_realdata):
        return np.mean(np.power((Y_realdata-self.Y_predict),2))
    
    def back_propogation(self,X,Y_realdata):
        dloss=-2*(Y_realdata-self.Y_predict) #derivative of L w.r.t P
        dpdn=np.ones_like(self.Y_predict) #derivative of P w.r.t N
        dpdb=np.ones_like(self.bias) #derivative of addition of bias ie P (A) w.r.t to bias
        dldn=dloss*dpdn #applying chain rule to get loss w.r.t to N (which is the dot productt)
        dndw=np.transpose(X,(1,0))

        dldw=np.dot(dndw,dldn)
        dldb = np.sum(dloss * dpdb, axis=0)

        loss_gradients = {}
        loss_gradients['w']=dldw
        loss_gradients['b']=dldb
        return loss_gradients
    
    def update_parameters(self,loss_gradients):
        self.weight -= self.learning_rate*loss_gradients['w']
        self.bias -= self.learning_rate*loss_gradients['b']
        
    def training(self,X,Y_realdata,epochs=20000):
        for epoch in range(epochs):
            #forward propogation
            Y_predict = self.forward_pass(X)
            #loss
            loss=self.compute_loss(Y_realdata)
            #backpropogation
            loss_gradients = self.back_propogation(X,Y_realdata)
            #update parameters
            self.update_parameters(loss_gradients)

            if epoch % 100 == 0 :
                print(f"Epoch {epoch}, Loss: {loss}")

def sigmoid(x):
    return 1/(1+ np.exp(-x))

#and dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y_realdata = np.array([0,1,1,1])

model = and_neural_model(2,0.1)
model.forward_pass(X)
loss = model.back_propogation(X,Y_realdata)
model.training(X,Y_realdata)
print("trained weight: ",model.weight)
print("trained bias",model.bias)

# newdata = np.array([[0,0],[0,0],[1,0],[1,1]]

while True:
    try:
        q= input("enter the first input of and: ")
        if q not in ['0','1']:
            print("invalid input")
            continue


        r= input("enter the second value of and: ")
        if r not in ['0','1']:
            print("invalid input")
            continue
        newdata = np.array([int(q),int(r)])
        predictions = model.forward_pass(newdata)
        result=np.round(predictions)
        print("the output of xor is: ",result)
    except ValueError: 
        print("error")