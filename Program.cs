using NeuralNetwork;

const int dataVectorCount = 3;

var mlp = new MultiLayerPerceptron(dataVectorCount, new [] {4,4,1 });

const int totalItemsInBatch = 4;

var xs = new double[ , ] {
  {2.0, 3.0, -1.0},
  {3.0, -1.0, 0.5},
  {0.5, 1.0, 1.0},
  {1.0, 1.0, -1.0},
};

var ys = new double [totalItemsInBatch] {1.0, -1.0, -1.0, 1.0};

var itemsToTrain = new List<List<Value>>();
var truth = ys.Select(x => new Value(x));

for(int i = 0; i < totalItemsInBatch; i++){
    var list = new List<Value>();
    for(int j = 0; j < dataVectorCount; j++){
        var z = xs[i,j];
        list.Add(new Value(z));
    }
    itemsToTrain.Add(list);
}

foreach(var epoch in Enumerable.Range(0,20)){
    var batchResult = itemsToTrain.Select( x => mlp.Call(x.ToArray()).First() );
    var loss = Enumerable.Zip(batchResult, truth).Select(x => (x.First - x.Second).Pow(2)).Sum();

    mlp.ZeroGradient();
    loss.Backward();
    mlp.Step();
    Console.WriteLine($"Epoch: {epoch}, loss: {loss.Data}");
}

