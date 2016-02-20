using Accord.Imaging;
using Accord.MachineLearning;
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace AccordNETLearning
{
    enum ClassType
    {
        Mario = 0,
        Bowser = 1
    }

    class BagOfWords
    {
        internal int NumberOfWords { get; private set; }

        public BagOfWords(int numberOfWords)
        {
            NumberOfWords = numberOfWords;
        }

        //use freak
        public List<FeatureSet> ComputeBagOfWords(params string[] imageDirectories)
        {
            KModes<byte> kmodes = new KModes<byte>(NumberOfWords, Distance.BitwiseHamming);
            
            FastRetinaKeypointDetector detector = new FastRetinaKeypointDetector();

            //create bow with the given algo
            BagOfVisualWords<FastRetinaKeypoint, byte[]> freakBow = new BagOfVisualWords<FastRetinaKeypoint, byte[]>(detector, kmodes);

            List<FeatureSet> computedFeatures = new List<FeatureSet>();

            foreach (var dirPath in imageDirectories)
            {
                Console.WriteLine("Enumerating files in: " + dirPath);
                var files = Directory.EnumerateFiles(dirPath, "*.png");
                
                //compute the bow codebook using training images only
                var images = (from f in files
                              select new Bitmap(f)).ToList().ToArray();

                Console.WriteLine("Computing BoW model...");
                freakBow.Compute(images);
                Console.WriteLine("BoW model complete.");

                foreach (var file in files)
                {
                    using (var image = new Bitmap(file))
                    {
                        //process image
                        Console.WriteLine("Getting feature vector for: " + file);
                        double[] featureVector = freakBow.GetFeatureVector(image);
                        ClassType type;
                        if(file.Contains("mario"))
                        {
                            type = ClassType.Mario;
                        }
                        else
                        {
                            type = ClassType.Bowser;
                        }
                        computedFeatures.Add(new FeatureSet(featureVector, type, Path.GetFileNameWithoutExtension(file)));
                    }
                }
            }

            return computedFeatures;
        }
    }
}
