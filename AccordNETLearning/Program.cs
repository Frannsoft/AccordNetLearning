using System;
using System.Collections.Generic;
using System.IO;

namespace AccordNETLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] trainingDirectories = new string[]
            {
                @"E:\train\mario",
                @"E:\train\bowser",
            };

            Console.WriteLine("Enter testing images path: ");
            string testingDirectoryPath = @"E:\test"; 

            var bow = new BagOfWords(36);
            List<FeatureSet> trainingFeatureSets = bow.ComputeBagOfWords(trainingDirectories);

            var svmHelper = new SVMHelper();
            var ksvm = svmHelper.CreateSVM(trainingFeatureSets);

            var classifier = new Classifier();
            List<FeatureSet> testingFeatureSets = bow.ComputeBagOfWords(testingDirectoryPath);

            Console.WriteLine("Testing...");

            double correctPercent = classifier.Classify(testingFeatureSets, ksvm);
            Console.WriteLine("Percent correctly classified: " + correctPercent + "%");
            Console.ReadLine();
        }
    }
}
