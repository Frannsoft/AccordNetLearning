using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AccordNETLearning
{
    class SVMHelper
    {
        public MulticlassSupportVectorMachine CreateSVM(List<FeatureSet> featureSets)
        {
            double[][] inputs;
            int[] outputs;

            getData(out inputs, out outputs, featureSets);

            int classes = outputs.Distinct().Count();

            var kernel = getKernel();

            //create the multi class support vector machine using the chi-square kernel
            Console.WriteLine("Creating SVM...");
            MulticlassSupportVectorMachine ksvm = new MulticlassSupportVectorMachine(inputs[0].Length, kernel, classes);

            //create the learning algorithm using the machine and the training data
            Console.WriteLine("Configuring the algorithm that will be used to train the machine..");
            MulticlassSupportVectorLearning ml = new MulticlassSupportVectorLearning(ksvm, inputs, outputs);

            //extract training parameters from the interface
            double complexity = 1.00000;
            double tolerance = 0.01000;
            int cacheSize = 0;
            SelectionStrategy strategy = SelectionStrategy.Sequential;

            //configure the learning algorithm
            ml.Algorithm = (svm, classInputs, classOutputs, i, j) =>
            {
                return new SequentialMinimalOptimization(svm, classInputs, classOutputs)
                {
                    Complexity = complexity,
                    Tolerance = tolerance,
                    CacheSize = cacheSize,
                    Strategy = strategy,
                    UseComplexityHeuristic = true
                };
            };

            Console.WriteLine("Training machine...this may take a while.");
            double error = ml.Run(); //train the machines.  should take a while

            return ksvm;
        }

        private void getData(out double[][] inputs, out int[] outputs, List<FeatureSet> featureSets)
        {
            List<double[]> inputList = new List<double[]>();
            List<int> outputList = new List<int>();

            foreach (var featureset in featureSets)
            {
                //if(featureset.IsTraining)
                //{
                inputList.Add(featureset.FeatureVector);
                outputList.Add(Convert.ToInt32(featureset.Class));
                //}
            }

            inputs = inputList.ToArray();
            outputs = outputList.ToArray();
        }

        private IKernel getKernel()
        {
            return new ChiSquare();
        }
    }
}
