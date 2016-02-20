using Accord.MachineLearning.VectorMachines;
using System;
using System.Collections.Generic;

namespace AccordNETLearning
{
    class Classifier
    {
        public double Classify(List<FeatureSet> featureSets, MulticlassSupportVectorMachine ksvm)
        {
            int totalTesting = 0;
            int totalCorrect = 0;

            Console.WriteLine("Predicting classes...");

            foreach (var featureSet in featureSets)
            {
                double[] input = featureSet.FeatureVector;
                int expected = (int)featureSet.Class;

                //classify into one of the classes
                int actual = ksvm.Compute(input);

                //check if we did a correct classification

                if (expected == actual)
                {
                    //yes
                    Console.WriteLine("Correct!  Image was: " + featureSet.Class);
                    totalCorrect++;
                }
                else
                {
                    //no
                    Console.WriteLine("Wrong: Image was: " + featureSet.Class + " and " + ((ClassType)actual).ToString() + " was predicted");
                }
                totalTesting++;
            }

            Console.WriteLine("Total correctly predicted: " + totalCorrect);
            Console.WriteLine("Total INcorrectly predicted: " + (totalTesting - totalCorrect));
            Console.WriteLine("Total images tested: " + totalTesting);
            double percentCorrect = (Convert.ToDouble(totalCorrect) / Convert.ToDouble(totalTesting)) * 100.0;
            return percentCorrect;
        }
    }
}
