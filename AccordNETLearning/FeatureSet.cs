namespace AccordNETLearning
{
    class FeatureSet
    {
        public double[] FeatureVector { get; private set; }
        public ClassType Class { get; private set; }
        public string Name { get; private set; }

        public FeatureSet(double[] featureVector, ClassType classValue, string name)
        {
            FeatureVector = featureVector;
            Class = classValue;
            Name = name;
        }
    }
}
