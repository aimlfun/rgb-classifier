namespace RGBclassifier;

/// <summary>
/// Represents a red-green-blue cross classifier training data item.
/// </summary>
internal struct ClassifierTrainingData
{
    /// <summary>
    /// The X cartesian coordinate scaled 0..width as 0..1 
    /// </summary>
    internal double inputX; // 0..1

    /// <summary>
    /// The Y cartesian coordinate scaled 0..height as 0..1 
    /// </summary>
    internal double inputY; // 0..1

    /// <summary>
    /// %age confidence indicator output from neural network for red crosses.
    /// </summary>
    internal double outputRed; // 0..1

    /// <summary>
    /// %age confidence indicator output from neural network for green crosses.
    /// </summary>
    internal double outputGreen; // 0..1

    /// <summary>
    /// %age confidence indicator output from neural network for blue crosses.
    /// </summary>
    internal double outputBlue; // 0..1

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="red"></param>
    /// <param name="green"></param>
    /// <param name="blue"></param>
    internal ClassifierTrainingData(double x, double y, double red, double green, double blue)
    {
        inputX = x;
        inputY = y;
        outputRed = red;
        outputGreen = green;
        outputBlue = blue;
    }
}
