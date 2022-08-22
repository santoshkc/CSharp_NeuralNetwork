namespace NeuralNetwork;
internal static class OperatorOverload {
    public static Value Pow(this Value self, double other) {
        var z = new Value(other);
        var output =  new Value(Math.Pow(self.Data, z.Data), children: (self, null ), operation: "^");
        
        output.BackwardCall = () => {
            self.Gradient += other * Math.Pow(self.Data, other - 1) * output.Gradient;
        };
        return output;
    }

    public static Value Sum(this IEnumerable<Value> items)
    {
        return items.Aggregate((left, right) => left + right);
    }
}
internal class Value {
    public double Data {get; set;}
    public string Label { get; set;}
    public string Operation { get; init;}
    public double Gradient {get; set;}
    public HashSet<Value> Previous {get; init;} = new HashSet<Value>();
    public Action BackwardCall {get; internal set;}
    public Value(double data , (Value, Value)? children = null , string label = "", string operation = "") {
        Data = data;
        Label = label;
        Operation = operation;
        Gradient = 0.0d;

        if(children.HasValue){
            if(children.Value.Item1 != null)
                Previous.Add(children.Value.Item1);

            if(children.Value.Item2 != null)
                Previous.Add(children.Value.Item2);
        }
        BackwardCall = () => {};
    }

    public void Backward() {
        var topologicalNodes = new List<Value>() { };
        var visitedNodes = new HashSet<Value>() { };

        void BuildTopologicalGraph(Value value) {
            if(visitedNodes.Contains(value) == false) {
                visitedNodes.Add(value);
                foreach(var children in value.Previous){
                    BuildTopologicalGraph(children);
                }
                topologicalNodes.Add(value);
            }
        }

        BuildTopologicalGraph(this);

        this.Gradient = 1.0;
        foreach(var node in topologicalNodes.Reverse<Value>()) {
            node.BackwardCall();
        }
    }

    public Value Tanh() {
        var x = Math.Exp(2*this.Data);
        var z = (x - 1)/(x + 1);
        var output = new Value(z, label: "tanh", children: (this, null) );

        output.BackwardCall = () => {
            this.Gradient += (1 - z * z) * output.Gradient;
        };
        return output;
    }

    public static Value operator- (Value other) {
        return other * -1.0;
    }

    public static Value operator-(Value self, double other) {
        return self + (-other);
    }

    public static Value operator-(Value self, Value other) {
        return self + (-other);
    }

    public static Value operator* (Value self, double other) {
        var o = new Value(other);
        return self * o;
    }

    public static Value operator* (double other, Value self) {
        return self *other;
    }

    public static Value operator* (Value self, int other) {
        var o = new Value(other);
        return self * o;
    }

    public static Value operator+ (Value self, double other) {
        var o = new Value(other);
        return self + o;
    }

    public static Value operator+ (Value self, int other) {
        var o = new Value(other);
        return self + o;
    }

    public static Value operator* (Value self, Value other) {
        var output = new Value(self.Data * other.Data, (self, other), operation: "*" );

        output.BackwardCall = () => {
            self.Gradient += other.Data * output.Gradient;
            other.Gradient += self.Data * output.Gradient;
        };

        return output;
    }

    public static Value operator+ (Value self, Value other) {
        var output =  new Value(self.Data + other.Data, (self, other), operation: "+" );

        output.BackwardCall = () => {
            var outputNode = output;
            self.Gradient += 1.0 * outputNode.Gradient;
            other.Gradient += 1.0 * outputNode.Gradient;
        };
        return output;
    }

    public override string ToString() {
        return $"Value(Label: {this.Label}, data: {this.Data}, grad: {this.Gradient})";
    }
}

internal class Neuron {
    public Value[] Weights {get; init;}

    public Value Bias {get; init;} 

    public double ComputeRandom() {
        var random = new System.Random(Environment.TickCount);
        var r1 = -1.0 + random.NextDouble()*2;
        return r1;
    }

    public Value[] Parameters() {
        var l = new List<Value>(Weights);
        l.Append(Bias);
        return l.ToArray();
    }

    public Neuron(int totalInputs) {
        var random = ComputeRandom();
        Weights = Enumerable.Range(0, totalInputs).Select(x => new Value(ComputeRandom())).ToArray();
        Bias = new Value(random);
    }

    public Value Call(Value[] inputs) {

        var sum = Enumerable.Range(0, inputs.Length).Select( x => Weights[x] * inputs[x] ).Sum() + Bias;
        return sum.Tanh();
    }
}

internal class Layer {

    public Neuron[] Neurons {get; init;}
    public Layer(int totalInputs, int totalOutputs) {
        Neurons  = Enumerable.Range(0, totalOutputs)
            .Select(x => new Neuron(totalInputs)).ToArray();
    }

    public Value[] Parameters() {
        return this.Neurons.SelectMany( x => x.Parameters()).ToArray();
    }

    public Value[] Call(Value[] inputs) {
        return Neurons.Select( x => x.Call(inputs)).ToArray();
    }
}

internal class MultiLayerPerceptron {
    public Layer[] Layers  {get; init;}
    public MultiLayerPerceptron(int inputCount, int[] layerCounts) {

        var counts = new List<int> { inputCount};
        counts.AddRange(layerCounts);

        Layers = Enumerable.Range(0, layerCounts.Count())
            .Select( x => new Layer( counts[x], counts[x+1]) ).ToArray();
    }

    public Value[] Call(Value[] inputs) {
        foreach(var layer in Layers) {
            inputs = layer.Call(inputs);
        }
       return inputs;
    }

    public Value[] Parameters() {
        return Layers.SelectMany( x => x.Parameters()).ToArray();
    }

    public void ZeroGradient() {
        foreach(var param in this.Parameters()){
            param.Gradient = 0.0;
        }
    }

    public void Step(double step = 0.01){

        foreach(var param in this.Parameters()){
            param.Data += -step * param.Gradient;
        }
    }
}

