using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text.Json;

////////////////////////////////| NN Notes|/////////////////////////////////////
// - Might want to consider adding some basic indicators to the data
//   such as SMA 10/20 or Smoothed RSI. Could potentially build and
//   return these inside the NN through their own methods.
//
// - Could optimize the hyperparameters (learningRate, epochs, hiddenSize)
//   after I test the "brain injection" in Quantower.
//
// - First training session has descreasing error showing the network is
//   improving, but the improvement is only ~6% reduction in error 
//   (from 0.220 to 0.206). There is definitely some unecessary noise
//   interfering with the network's ability to recognize patterns.
//
// - LSTM seems to be the next logical step in this journey.
////////////////////////////////////////////////////////////////////////////////


class Program
{
    static void Main(string[] args)
    {
        // Path to CSV file and json output "brain" path
        string csvPath = @"C:es_data_2022_2024.csv";
        string modelPath = @"C:sp500_neural_network_model.json";

        // Initialize neural network: 50 inputs (10 bars * OHLCV), 32 hidden, 4 outputs
        NeuralNetwork nn = new NeuralNetwork(inputSize: 50, hiddenSize: 32, outputSize: 4, learningRate: 0.05);

        // Read and prepare data
        Console.WriteLine("Reading and preparing data...");
        var (inputs, targets) = PrepareData(csvPath);

        // Split data: 80% train, 20% test
        int trainSize = (int)(inputs.Count * 0.8);
        var trainInputs = inputs.Take(trainSize).ToList();
        var trainTargets = targets.Take(trainSize).ToList();
        var testInputs = inputs.Skip(trainSize).ToList();
        var testTargets = targets.Skip(trainSize).ToList();

        // Train the network
        Console.WriteLine("Training neural network...");
        nn.Train(trainInputs.ToArray(), trainTargets.ToArray(), epochs: 1000);

        // Test the network
        Console.WriteLine("\nTesting neural network...");
        double totalError = 0;
        for (int i = 0; i < testInputs.Count; i++)
        {
            double[] output = nn.Forward(testInputs[i]);
            double error = 0;
            for (int j = 0; j < output.Length; j++)
                error += Math.Pow(testTargets[i][j] - output[j], 2);
            totalError += error / output.Length;
            if (i < 5) // Print first 5 test results
                Console.WriteLine($"Test {i + 1}: Predicted [{string.Join(", ", output.Select(x => x.ToString("F4")))}], Expected [{string.Join(", ", testTargets[i])}]");
        }
        Console.WriteLine($"Average Test Error: {totalError / testInputs.Count:F6}");

        // Save the trained model
        Console.WriteLine($"Saving model to: {Path.GetFullPath(modelPath)}");
        nn.SaveModel(modelPath);
        Console.WriteLine($"\nModel saved to {modelPath}");
    }

    static (List<double[]>, List<double[]>) PrepareData(string csvPath)
    {
        List<double[]> inputs = new List<double[]>();
        List<double[]> targets = new List<double[]>();
        List<(DateTime dateTime, double open, double high, double low, double close, double volume)> bars = new List<(DateTime, double, double, double, double, double)>();

        // Read CSV file
        using (var reader = new StreamReader(csvPath))
        {
            string header = reader.ReadLine(); // Skip header
            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine();
                var parts = line.Split(',');
                if (parts.Length < 7) continue; // Skip malformed rows
                try
                {
                    DateTime dateTime = DateTime.Parse(parts[1]);
                    double open = double.Parse(parts[2]);
                    double high = double.Parse(parts[3]);
                    double low = double.Parse(parts[4]);
                    double close = double.Parse(parts[5]);
                    double volume = double.Parse(parts[6]);
                    bars.Add((dateTime, open, high, low, close, volume));
                }
                catch { continue; } // Skip rows with parsing errors
            }
        }

        // Generate inputs and targets
        const int lookbackBars = 10; // May want to train with 5 or 20 later to capture different time horizons
        for (int i = lookbackBars; i < bars.Count - 5; i++) // Reserve 5 bars for future price check
        {
            // Collect OHLCV for last 10 bars
            double[] input = new double[lookbackBars * 5];
            for (int j = 0; j < lookbackBars; j++)
            {
                var bar = bars[i - lookbackBars + j];
                input[j * 5 + 0] = bar.open;
                input[j * 5 + 1] = bar.high;
                input[j * 5 + 2] = bar.low;
                input[j * 5 + 3] = bar.close;
                input[j * 5 + 4] = bar.volume;
            }

            // Normalize inputs
            double minPrice = input.Take(lookbackBars * 4).Min();
            double maxPrice = input.Take(lookbackBars * 4).Max();
            double minVolume = input.Skip(lookbackBars * 4).Min();
            double maxVolume = input.Skip(lookbackBars * 4).Max();
            if (maxPrice == minPrice || maxVolume == minVolume) continue;
            for (int j = 0; j < lookbackBars * 4; j++)
                input[j] = (input[j] - minPrice) / (maxPrice - minPrice);
            for (int j = lookbackBars * 4; j < lookbackBars * 5; j++)
                input[j] = (input[j] - minVolume) / (maxVolume - minVolume);

            // Generate synthetic targets based on future price movement
            double currentClose = bars[i].close;
            double futureClose = bars[i + 5].close; // Look 5 minutes ahead
            double[] target = new double[4]; // [openBuy, closeBuy, openSell, closeSell]

            ///////////////| If Adding TP and SL |/////////////////////
            // Handle all entry/exit of positions in lines below, and
            // consider adding psuedo SL and TP
            ///////////////////////////////////////////////////////////

            // Open Buy: Price rises by 0.5% in 5 minutes
            if (futureClose >= currentClose * 1.005)
                target[0] = 1;

            // Set SL and TP for buy position here later

            // Close Buy: Assume position opened 10 bars ago, close if loss > 0.2% or after 10 bars
            if (i >= lookbackBars + 10)
            {
                double entryPrice = bars[i - 10].close;
                if (currentClose <= entryPrice * 0.998 || i % 10 == 0)
                    target[1] = 1;
            }

            // Open Sell: Price falls by 0.5% in 5 minutes
            if (futureClose <= currentClose * 0.995)
                target[2] = 1;

            // Set SL and TP for sell position here later

            // Close Sell: Assume position opened 10 bars ago, close if loss > 0.2% or after 10 bars
            if (i >= lookbackBars + 10)
            {
                double entryPrice = bars[i - 10].close;
                if (currentClose >= entryPrice * 1.002 || i % 10 == 0)
                    target[3] = 1;
            }

            inputs.Add(input);
            targets.Add(target);
        }

        return (inputs, targets);
    }

    class NeuralNetwork
    {
        private int inputSize; // Gets defined in Main(), fed into Forward()
        private int hiddenSize;
        private int outputSize;
        private double[,] weightsInputHidden;
        private double[,] weightsHiddenOutput;
        private double[] biasHidden;
        private double[] biasOutput;
        private Random rand;
        private double learningRate;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate = 0.1)
        {
            this.inputSize = inputSize; // Gets defined in Main(), fed into Forward()
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.learningRate = learningRate;
            rand = new Random();

            weightsInputHidden = new double[inputSize, hiddenSize];
            weightsHiddenOutput = new double[hiddenSize, outputSize];
            biasHidden = new double[hiddenSize];
            biasOutput = new double[outputSize];

            InitializeWeights();
        }


        private void InitializeWeights() // Sets initial values for weights and biases before training
        {
            // Creates connections between 50 OHLCV inputs and 32 hidden neurons
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    weightsInputHidden[i, j] = rand.NextDouble() * 0.2 - 0.1;
            // Links 32 hidden neurons to 4 trading signal outputs (open/close buy/sell)
            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < outputSize; j++)
                    weightsHiddenOutput[i, j] = rand.NextDouble() * 0.2 - 0.1;
            // Provides an offset for each hidden neuron's activation (produces non-zero outputs)
            for (int i = 0; i < hiddenSize; i++)
                biasHidden[i] = rand.NextDouble() * 0.2 - 0.1;
            // Provides an offset for each output neuron's prediction (produces non-zero outputs)
            for (int i = 0; i < outputSize; i++)
                biasOutput[i] = rand.NextDouble() * 0.2 - 0.1;
        }

        // Beautiful lambda expressions, reference Discord chat if need to rework
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        private double SigmoidDerivative(double x) => x * (1.0 - x);


        ///////////////////| Forward() Function Breakdown|//////////////////////////
        // This method is the forward pass of my NN. It's taking an input vector
        // and computing the network's predictions by "propagating the input
        // through the layers". Effectively the heart of my neural network's
        // prediction process. This is transforming the market data into
        // trading signals. Would need to be reworked with "gates" to adapt to
        // LSTM, which is my most likely next move.
        ////////////////////////////////////////////////////////////////////////////
        public double[] Forward(double[] inputs)
        {
            if (inputs.Length != inputSize)
                throw new ArgumentException("Input size mismatch");

            double[] hidden = new double[hiddenSize]; // Array to store the activations of the
                                                      // hidden layer's neurons (==hiddenSize)
            
            for (int j = 0; j < hiddenSize; j++) // Computing activation for each hidden neuron
            {
                double sum = biasHidden[j];
                for (int i = 0; i < inputSize; i++)
                    sum += inputs[i] * weightsInputHidden[i, j]; // Is this right? Check again (yes)
                hidden[j] = Sigmoid(sum);
            }

            double[] outputs = new double[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                double sum = biasOutput[j];
                for (int i = 0; i < hiddenSize; i++)
                    sum += hidden[i] * weightsHiddenOutput[i, j];
                outputs[j] = Sigmoid(sum);
            }

            return outputs; // Literally most important return. Provides final predictions for use in
                            // training, testing, and actively trading. Goes to the caller.
        }

        ///////////////////////////| Train() Function Breakdown|/////////////////////////////////
        // Uses gradient descent to minimize MSE (mean squared error) by iteratively performing
        // forward and back passes. Trains the feedforward neural network with backpropagation.
        // Should be adjusting the network's parameters to minimize the error between predicted
        // and target outputs. Considering I copy-pasted this function and barely made
        // adjustments, I am still having a hard time understanding a lot of this. My initial
        // training function was absolute dogwater. The indices variable should be randomly
        // shuffling input-target pairs to prevent learning order-specific patterns. Doing so
        // should improve generalization (basically, making the NN better at performing well
        // on unseen data when trading with a live data environment).
        //////////////////////////////////////////////////////////////////////////////////////////
        public void Train(double[][] inputs, double[][] targets, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;
                // Indices is shuffling data for better training, need to print debug later in order to 
                // understand (or maybe just pass off to an LLM and have the LLM break it down for me)
                var indices = Enumerable.Range(0, inputs.Length).OrderBy(_ => rand.NextDouble()).ToArray();
                for (int idx = 0; idx < inputs.Length; idx++)
                {
                    int i = indices[idx];
                    double[] input = inputs[i];
                    double[] hidden = new double[hiddenSize];
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        double sum = biasHidden[j];
                        for (int k = 0; k < inputSize; k++)
                            sum += input[k] * weightsInputHidden[k, j];
                        hidden[j] = Sigmoid(sum);
                    }

                    double[] outputs = new double[outputSize];
                    for (int j = 0; j < outputSize; j++)
                    {
                        double sum = biasOutput[j];
                        for (int k = 0; k < hiddenSize; k++)
                            sum += hidden[k] * weightsHiddenOutput[k, j];
                        outputs[j] = Sigmoid(sum);
                    }

                    double[] outputErrors = new double[outputSize];
                    for (int j = 0; j < outputSize; j++)
                    {
                        outputErrors[j] = targets[i][j] - outputs[j];
                        totalError += Math.Pow(outputErrors[j], 2);
                    }

                    double[] hiddenErrors = new double[hiddenSize];
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        hiddenErrors[j] = 0;
                        for (int k = 0; k < outputSize; k++)
                            hiddenErrors[j] += outputErrors[k] * weightsHiddenOutput[j, k] * SigmoidDerivative(outputs[k]);
                    }

                    for (int j = 0; j < outputSize; j++)
                    {
                        biasOutput[j] += learningRate * outputErrors[j] * SigmoidDerivative(outputs[j]);
                        for (int k = 0; k < hiddenSize; k++)
                            weightsHiddenOutput[k, j] += learningRate * outputErrors[j] * hidden[k] * SigmoidDerivative(outputs[j]);
                    }

                    for (int j = 0; j < hiddenSize; j++)
                    {
                        biasHidden[j] += learningRate * hiddenErrors[j] * SigmoidDerivative(hidden[j]);
                        for (int k = 0; k < inputSize; k++)
                            weightsInputHidden[k, j] += learningRate * hiddenErrors[j] * input[k] * SigmoidDerivative(hidden[j]);
                    }
                }

                if (epoch % 100 == 0)
                    Console.WriteLine($"Epoch {epoch}, Error: {totalError / inputs.Length:F6}");
            }
        }

        public void SaveModel(string path) // Kept failing, then suddenly didn't. Why.
        {
            // Convert multidimensional arrays to jagged arrays for serialization
            double[][] jaggedWeightsInputHidden = new double[inputSize][];
            for (int i = 0; i < inputSize; i++)
            {
                jaggedWeightsInputHidden[i] = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++)
                    jaggedWeightsInputHidden[i][j] = weightsInputHidden[i, j];
            }

            double[][] jaggedWeightsHiddenOutput = new double[hiddenSize][];
            for (int i = 0; i < hiddenSize; i++)
            {
                jaggedWeightsHiddenOutput[i] = new double[outputSize];
                for (int j = 0; j < outputSize; j++)
                    jaggedWeightsHiddenOutput[i][j] = weightsHiddenOutput[i, j];
            }

            var model = new
            {
                WeightsInputHidden = jaggedWeightsInputHidden,
                WeightsHiddenOutput = jaggedWeightsHiddenOutput,
                BiasHidden = biasHidden,
                BiasOutput = biasOutput
            };
            string json = JsonSerializer.Serialize(model, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, json);
        }

        public void LoadModel(string path) // Needed for Quantower "brain" activation
        {
            string json = File.ReadAllText(path);
            var model = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);

            // Convert jagged arrays back to multidimensional arrays
            var jaggedWeightsInputHidden = model["WeightsInputHidden"].Deserialize<double[][]>();
            weightsInputHidden = new double[inputSize, hiddenSize];
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    weightsInputHidden[i, j] = jaggedWeightsInputHidden[i][j];

            var jaggedWeightsHiddenOutput = model["WeightsHiddenOutput"].Deserialize<double[][]>();
            weightsHiddenOutput = new double[hiddenSize, outputSize];
            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < outputSize; j++)
                    weightsHiddenOutput[i, j] = jaggedWeightsHiddenOutput[i][j];

            biasHidden = model["BiasHidden"].Deserialize<double[]>();
            biasOutput = model["BiasOutput"].Deserialize<double[]>();
        }
    }
}