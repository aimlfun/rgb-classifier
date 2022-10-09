using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Security.Cryptography;

namespace RGBclassifier;


/// <summary>
/// Demo of classification: 
/// Inputs: x, y are classed as red, green or blue.
/// Output: visualisation of how the AI learns to make boundaries around the inputs.
/// </summary>
public partial class FormCrosses : Form
{
    /// <summary>
    /// Shape has green crosses within red/blue-cross rectangles. Setting to true will take lot of epoch's to find a perfect fit.
    /// It might not even manage to.
    /// </summary>
    const bool c_complexShape = false;

    /// <summary>
    /// true - red box and blue box are computed.
    /// false - wiggly line is drawn with red one side of it, green the other.
    /// </summary>
    const bool c_colouredBoxes = true; 

    /// <summary>
    /// If the neural network output is greater than this, then it is considered matching.
    /// i.e. at 0.6, if the "B" output returns 0.7, then it assumes "blue".
    /// i.e. at 0.6, if the "B" output returns 0.5, then it doesn't think it should be "blue".
    /// </summary>
    const float c_thresholdRequiredToConsiderAIhasChosenThisColour = 0.6f; //0..1 => 0%->100%. Anything lower than 0.5 isn't good confidence level.

    /// <summary>
    /// Determines amount of error before it draws a circle around a cross.
    /// </summary>
    const float c_marginOfErrorToCircle = 0.6f;

    /// <summary>
    /// The neural network
    /// </summary>
    private readonly NeuralNetwork network;

    /// <summary>
    /// Location of crosses split by colour.
    /// </summary>
    private readonly Dictionary<string, List<Point>> pointsCrossKeyedByColour = new();

    /// <summary>
    /// Data containing crosses, to train.
    /// </summary>
    private readonly List<ClassifierTrainingData> trainingData = new();

    /// <summary>
    /// Width of the picture box.
    /// </summary>
    private readonly int width;

    /// <summary>
    /// Height of the picture box.
    /// </summary>
    private readonly int height;

    /// <summary>
    /// Epoch is the generation.
    /// </summary>
    private int epoch = 0;

    #region FORM EVENTS
    /// <summary>
    /// Constructor.
    /// </summary>
    public FormCrosses()
    {
        InitializeComponent();

        int[] AIHiddenLayers = new int[] { 2, 8, 8, 3 }; // 2 inputs (x,y) 2x8 hidden, 3 outputs (1=red,1=green,1=blue)

        ActivationFunctions[] AIactivationFunctions = new ActivationFunctions[] { ActivationFunctions.TanH,  // 2
                                                                                  ActivationFunctions.TanH,  // 8
                                                                                  ActivationFunctions.TanH,  // 8 
                                                                                  ActivationFunctions.TanH}; // 3

        network = new NeuralNetwork(AIHiddenLayers, AIactivationFunctions);

        // weirdly access picturebox/image height or width is slow. It's like it has to calculate it.
        height = pictureBoxCrosses.Height;
        width = pictureBoxCrosses.Width;

        // the idea is to put a sea of green crosses, and have red crosses in a small part
        if (c_colouredBoxes) GenerateRandomRedGreenCrosses(); else GenerateRandomRedGreenCrosses2();

        labelConfidence.Text = $"{Math.Round(c_thresholdRequiredToConsiderAIhasChosenThisColour * 100, 2)}%";
    }

    /// <summary>
    /// On load, we make training data out of the crosses. A timer is started to backpropagate.
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void Form1_Load(object sender, EventArgs e)
    {
        PlotTheColouredCrosses();
        CreateTrainingData();

        timerTrainAndPlot.Tick += TimerTrainAndPlot_Tick;
        timerTrainAndPlot.Start();
    }

    /// <summary>
    /// Timer fires and initiates a further training attempt then plots output.
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void TimerTrainAndPlot_Tick(object? sender, EventArgs e)
    {
        Train();

        labelEpoch.Text = $"Epoch {++epoch}";

        if (epoch % 10 != 0) return; // only paint every 10 trainings. Stubbing it out works, but slower (as painting is expensive)

        int step = 6;

        List<Point> redPointsIdentifiedByNeuralNetwork = new();
        List<Point> bluePointsIdentifiedByNeuralNetwork = new();
        List<int[]> heatMapOfNeuralNetworkOutputForEveryPoint = new();

        // move across the space and if the AI is confident, add the square to a list to paint
        for (int x = 0; x < width; x += step)
        {
            for (int y = 0; y < height; y += step)
            {
                // we don't want to use "x" and "y" in raw form, as they will squashed by the activation function to +/-1, so we
                // scale them e.g. 1=width, 0=none, 0.5=middle of width.
                double[] result = network.FeedForward(inputs: new double[] { (float)x / width, (float)y / height });

                if (result[0] > c_thresholdRequiredToConsiderAIhasChosenThisColour) redPointsIdentifiedByNeuralNetwork.Add(new Point(x, y));

                // we don't do result[1], the green is implied if it isn't red/blue

                if (result[2] > c_thresholdRequiredToConsiderAIhasChosenThisColour) bluePointsIdentifiedByNeuralNetwork.Add(new Point(x, y));

                heatMapOfNeuralNetworkOutputForEveryPoint.Add(new int[] { x, y, ScaleFloat0to1Into0to255(result[0]),    // R 
                                                                                ScaleFloat0to1Into0to255(result[1]),    // G
                                                                                ScaleFloat0to1Into0to255(result[2]) }); // B
            }
        }

        // this plots a heatmap of every point in R/G/B
        PlotHeatMapOfNeuralNetworkOutput(heatMapOfNeuralNetworkOutputForEveryPoint);
        List<ClassifierTrainingData> pointsRequiringACircleWhereNeuralOutputIsIncorrectByMoreThanThreshold = CalculateWhichCirclesIndicatingErrorPct();

        PlotCrossesWithOverlayOfThoseIdentifiedByNeuralNetwork(redPointsIdentifiedByNeuralNetwork, bluePointsIdentifiedByNeuralNetwork, pointsRequiringACircleWhereNeuralOutputIsIncorrectByMoreThanThreshold);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    private List<ClassifierTrainingData> CalculateWhichCirclesIndicatingErrorPct()
    {
        List<ClassifierTrainingData> pointsRequiringACircleWhereNeuralOutputIsIncorrectByMoreThanThreshold = new();

        foreach (ClassifierTrainingData data in trainingData)
        {
            network.BackPropagate(inputs: new double[] { data.inputX, data.inputY },
                                   expected: new double[] { data.outputRed, data.outputGreen, data.outputGreen });

            double[] result = network.FeedForward(inputs: new double[] { data.inputX, data.inputY });

            double errorForRedCross = Math.Abs(data.outputRed - result[0]);
            double errorForGreenCross = Math.Abs(data.outputGreen - result[1]);
            double errorForBlueCross = Math.Abs(data.outputBlue - result[2]);

            // if points are wrongly classified (exceeds an error margin) we put a circle on them
            if (errorForRedCross > c_marginOfErrorToCircle ||
                errorForGreenCross > c_marginOfErrorToCircle ||
                errorForBlueCross > c_marginOfErrorToCircle)
            {
                pointsRequiringACircleWhereNeuralOutputIsIncorrectByMoreThanThreshold.Add(new ClassifierTrainingData(x: data.inputX,
                                                                                                                    y: data.inputY,
                                                                                                                    red: errorForRedCross > c_marginOfErrorToCircle ? errorForRedCross : 0,
                                                                                                                    green: errorForGreenCross > c_marginOfErrorToCircle ? errorForGreenCross : 0,
                                                                                                                    blue: errorForBlueCross > c_marginOfErrorToCircle ? errorForBlueCross : 0));
            }
        }

        return pointsRequiringACircleWhereNeuralOutputIsIncorrectByMoreThanThreshold;
    }
    #endregion

    #region TRAINING
    /// <summary>
    /// Add the crosses as training data.
    /// </summary>
    private void CreateTrainingData()
    {
        // for every red cross,   we add x,y, r=1,g=0,b=0
        // for every green cross, we add x,y, r=0,g=1,b=0
        // for every blue cross,  we add x,y, r=0,g=0,b=1

        // The neural network associates that x,y with one of 3 outputs. i.e. classify x,y

        // define training data based on points
        foreach (Point p in pointsCrossKeyedByColour["red"])
            trainingData.Add(new ClassifierTrainingData(x: (float)p.X / width, y: (float)p.Y / height, red: 1, green: 0, blue: 0));

        foreach (Point p in pointsCrossKeyedByColour["green"])
            trainingData.Add(new ClassifierTrainingData(x: (float)p.X / width, y: (float)p.Y / height, red: 0, green: 1, blue: 0));

        foreach (Point p in pointsCrossKeyedByColour["blue"])
            trainingData.Add(new ClassifierTrainingData(x: (float)p.X / width, y: (float)p.Y / height, red: 0, green: 0, blue: 1));
    }

    /// <summary>
    /// Train the neural network in a random order using training data.
    /// </summary>
    private void Train()
    {
        Random randomPRNG = new(); // we don't need crypto rng for this

        int traingDataPoints = trainingData.Count;

        // train using random points, rather than sequential.
        // WHY? sequential in some circumstances ends up with an unstable back-propagation. 
        for (int i = 0; i < traingDataPoints; i++)
        {
            int indexOfTrainingItem = randomPRNG.Next(0, trainingData.Count); // train on random

            ClassifierTrainingData d = trainingData[indexOfTrainingItem]; // 5 elements X,Y, R,G,B

            network.BackPropagate(inputs: new double[] { d.inputX, d.inputY },
                                  expected: new double[] { d.outputRed, d.outputGreen, d.outputBlue });
        }
    }
    #endregion

    /// <summary>
    /// Randomly creates a rectangle (40x40px minimum) within the width/height space.
    /// </summary>
    /// <returns>A random rectangle.</returns>
    private Rectangle GetRandomRectangle()
    {
        // pick a random place
        int corner1X = RandomNumberGenerator.GetInt32(30, width - 30);
        int corner1Y = RandomNumberGenerator.GetInt32(30, height - 30);

        // pick a random place
        int corner2X = RandomNumberGenerator.GetInt32(30, width - 30);
        int corner2Y = RandomNumberGenerator.GetInt32(30, height - 30);

        // ensure it is a minimum size. This adjustment can move them crosses off screen.
        if (Math.Abs(corner1X - corner2X) < 100)
        {
            corner1X -= 50;
            corner2X += 50;
        }

        if (Math.Abs(corner1Y - corner2Y) < 100)
        {
            corner1Y -= 50;
            corner2Y += 50;
        }

        // to compare if a point is in the "box" for red crosses, we need to order the top/left-bottom/right coordinates.
        // we also need them within the screen area.
        int minX = Math.Min(corner1X, corner2X).Clamp(0, width);
        int minY = Math.Min(corner1Y, corner2Y).Clamp(0, height);

        int maxX = Math.Max(corner1X, corner2X).Clamp(0, width);
        int maxY = Math.Max(corner1Y, corner2Y).Clamp(0, height);

        return Rectangle.FromLTRB(minX, minY, maxX, maxY);
    }

    /// <summary>
    /// Fills pointsRedCrosses/pointsGreenCrosses with the location of the crosses (indicating 1/0)
    /// </summary>
    private void GenerateRandomRedGreenCrosses()
    {
        // pick 2 coloured rectangles
        Rectangle redRectangle = GetRandomRectangle();
        Rectangle blueRectangle = GetRandomRectangle();

        // initialise in case the random crosses results in none of a particular colour (avoids a crash looking up non existent dictionary key)
        pointsCrossKeyedByColour.Add("red", new());
        pointsCrossKeyedByColour.Add("blue", new());
        pointsCrossKeyedByColour.Add("green", new());

        int step = RandomNumberGenerator.GetInt32(8, 15);

        // randomly add the crosses
        for (int x = 0; x < width; x += step)
        {
            for (int y = 0; y < height; y += step)
            {
                if (RandomNumberGenerator.GetInt32(0, 100) < 40)
                {
                    if (redRectangle.Contains(x, y))
                        AddCross("red", new Point(x, y));
                    else
                        if (blueRectangle.Contains(x, y)) AddCross("blue", new Point(x, y));
                    else
                        AddCross("green", new Point(x, y));
                }
                else
#pragma warning disable CS0162 // Unreachable code detected: Technically correct, but intentionally conditional code. Code below is used if constant c_complexShape==true.
                    if (c_complexShape) AddCross("green", new Point(x, y));
#pragma warning restore CS0162 // Unreachable code detected
            }
        }
    }

    /// <summary>
    /// Fills pointsRedCrosses/pointsGreenCrosses with the location of the crosses (indicating 1/0).
    /// This makes a random wiggly line splitting red and green.
    /// </summary>
    private void GenerateRandomRedGreenCrosses2()
    {
        pointsCrossKeyedByColour.Add("red", new());
        pointsCrossKeyedByColour.Add("green", new());
        pointsCrossKeyedByColour.Add("blue", new());

        // to compare if a point is in the "box" for red crosses, we need to order the top/left-bottom/right coordinates.
        int step;

        float y = height / 2;
        float x = 0;
        float angle = 0;
        float pixelSize = 6;

        angle += ((float)RandomNumberGenerator.GetInt32(-15, 15));

        // add the crosses
        while (x < width)
        {
            if (RandomNumberGenerator.GetInt32(0, 100) < 30)
            {
                angle += ((float)RandomNumberGenerator.GetInt32(-15, 15));
                angle = angle.Clamp(-45, 45);
            }

            double angleInRadians = MathUtils.DegreesInRadians(angle);

            step = RandomNumberGenerator.GetInt32(1, 5);
            y += (int)Math.Round(Math.Sin(angleInRadians) * step * 8);
            y = y.Clamp(0, height);
            x += (int)Math.Round(Math.Cos(angleInRadians) * step) + 15;

            for (float z = y - 5; z > 0; z -= RandomNumberGenerator.GetInt32(27, 37))
            {
                float px = x + RandomNumberGenerator.GetInt32(-5, 5);
                float py = z;

                px = ((int)(px / pixelSize)) * pixelSize - 10;
                py = ((int)(py / pixelSize)) * pixelSize;

                AddCross("green", new Point((int)px, (int)py));
            }

            for (float z = y + 5; z < height; z += RandomNumberGenerator.GetInt32(27, 37))
            {
                float px = x + RandomNumberGenerator.GetInt32(-5, 5);
                float py = z;

                px = ((int)(px / pixelSize)) * pixelSize - 10;
                py = ((int)(py / pixelSize)) * pixelSize;

                AddCross("red", new Point((int)px, (int)py));
            }
        }
    }

    /// <summary>
    /// Safely adds a cross to our dictionary of coloured crosses.
    /// </summary>
    /// <param name="colour"></param>
    /// <param name="pointOfCross"></param>
    private void AddCross(string colour, Point pointOfCross)
    {
        if (!pointsCrossKeyedByColour.ContainsKey(colour)) pointsCrossKeyedByColour.Add(colour, new());

        pointsCrossKeyedByColour[colour].Add(pointOfCross);
    }

    /// <summary>
    /// Plots the red, green and blue crosses.
    /// </summary>
    private void PlotTheColouredCrosses()
    {
        Bitmap bitmap = new(pictureBoxCrosses.Width, pictureBoxCrosses.Height);

        using Graphics graphics = Graphics.FromImage(bitmap);
        graphics.Clear(Color.Black);
        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

        foreach (Point point in pointsCrossKeyedByColour["red"]) DrawX(graphics, point, Pens.Red);
        foreach (Point point in pointsCrossKeyedByColour["blue"]) DrawX(graphics, point, Pens.Blue);
        foreach (Point point in pointsCrossKeyedByColour["green"]) DrawX(graphics, point, Pens.Green);

        graphics.Flush();

        pictureBoxCrosses.Image?.Dispose();
        pictureBoxCrosses.Image = bitmap;
    }

    /// <summary>
    /// Draws the right hand image with crosses overlaid with where the NN has classified as well as error circles.
    /// </summary>
    /// <param name="pointsRedCrossesToPlot"></param>
    /// <param name="pointsBlueCrossesToPlot"></param>
    /// <param name="pointsRequiringACircleWhereNeuralOutputIsIncorrectByMoreThanThreshold"></param>
    private void PlotCrossesWithOverlayOfThoseIdentifiedByNeuralNetwork(List<Point> pointsRedCrossesToPlot, List<Point> pointsBlueCrossesToPlot, List<ClassifierTrainingData> pointsRequiringACircleWhereNeuralOutputIsIncorrectByMoreThanThreshold)
    {
        Bitmap bitmap = new(pictureBoxCrosses.Image);

        using Graphics graphics = Graphics.FromImage(bitmap);
        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

        using SolidBrush redBrush = new(Color.FromArgb(80, 255, 150, 150));
        foreach (Point point in pointsRedCrossesToPlot) Draw6x6RectangleAtPositionUsingBrush(graphics, point, redBrush);

        // we don't overlay green. We could, but it's more painting and you can see from the heatmap the rest is mostly green.

        using SolidBrush blueBrush = new(Color.FromArgb(80, 150, 150, 255));
        foreach (Point point in pointsBlueCrossesToPlot) Draw6x6RectangleAtPositionUsingBrush(graphics, point, blueBrush);

        using SolidBrush brush = new(Color.Black);

        // X,Y,R,G,B
        // 0 1 2 3 4
        foreach (ClassifierTrainingData trainingData in pointsRequiringACircleWhereNeuralOutputIsIncorrectByMoreThanThreshold)
        {
            // black? don't plot
            if (Math.Abs(trainingData.outputRed) +
                Math.Abs(trainingData.outputGreen) +
                Math.Abs(trainingData.outputBlue) == 0) continue;

            brush.Color = Color.FromArgb(
                                            alpha: (ScaleFloat0to1Into0to255(trainingData.outputRed + trainingData.outputGreen + trainingData.outputBlue) / 3).Clamp(0, 255),
                                            red: ScaleFloat0to1Into0to255(trainingData.outputRed),
                                            green: ScaleFloat0to1Into0to255(trainingData.outputGreen),
                                            blue: ScaleFloat0to1Into0to255(trainingData.outputBlue));

            Draw8x8CircleAtPosition(graphics: graphics,
                                    position: new((int)(trainingData.inputX * width), (int)(trainingData.inputY * height)),
                                    brush: brush);
        }

        graphics.Flush();

        pictureBoxOverlayOfOutput.Image?.Dispose();
        pictureBoxOverlayOfOutput.Image = bitmap;
    }

    /// <summary>
    /// Clamp 0..1, and scale to 0..255.
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    private static int ScaleFloat0to1Into0to255(double value)
    {
        return Math.Abs((int)(255 * value)).Clamp(0, 255);
    }

    /// <summary>
    /// Colours the output of the neural network.
    /// </summary>
    /// <param name="mapOfNeuralNetworkOutput"></param>
    private void PlotHeatMapOfNeuralNetworkOutput(List<int[]> mapOfNeuralNetworkOutput)
    {
        Bitmap bitmap = new(pictureBoxNeuralNetworkOutput.Width, pictureBoxNeuralNetworkOutput.Height);

        using Graphics graphics = Graphics.FromImage(bitmap);
        graphics.Clear(Color.Black);
        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

        using SolidBrush brush = new(Color.Black); // rather than create a brush each time, we change the color

        foreach (int[] data in mapOfNeuralNetworkOutput)
        {
            brush.Color = Color.FromArgb((data[2] + data[3] + data[4]) / 3, data[2], data[3], data[4]);
            Draw6x6RectangleAtPositionUsingBrush(graphics, new Point(data[0], data[1]), brush);
        }

        graphics.Flush();

        pictureBoxNeuralNetworkOutput.Image?.Dispose();
        pictureBoxNeuralNetworkOutput.Image = bitmap;
    }

    /// <summary>
    /// Draws a red, green or blue "x".
    /// </summary>
    /// <param name="graphics"></param>
    /// <param name="position"></param>
    /// <param name="pen"></param>
    private static void DrawX(Graphics graphics, Point position, Pen pen)
    {
        // x marks the spot for center of mass
        graphics.DrawLine(pen, position.X - 2, position.Y - 2, position.X + 2, position.Y + 2);
        graphics.DrawLine(pen, position.X - 2, position.Y + 2, position.X + 2, position.Y - 2);
    }

    /// <summary>
    /// Draws a rectangle at the position.
    /// </summary>
    /// <param name="graphics"></param>
    /// <param name="position"></param>
    /// <param name="brush"></param>
    private static void Draw6x6RectangleAtPositionUsingBrush(Graphics graphics, Point position, Brush brush)
    {
        graphics.FillRectangle(brush, position.X - 3, position.Y - 3, 6, 6);
    }

    /// <summary>
    /// Draws a circle at the position.
    /// </summary>
    /// <param name="graphics"></param>
    /// <param name="position"></param>
    /// <param name="brush"></param>
    private static void Draw8x8CircleAtPosition(Graphics graphics, Point position, Brush brush)
    {
        // x marks the spot for center of mass
        graphics.FillEllipse(brush, position.X - 4, position.Y - 4, 8, 8);
    }

    /// <summary>
    /// Provide a capability pto pause.
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void FormCrosses_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.P) timerTrainAndPlot.Enabled = !timerTrainAndPlot.Enabled;
    }
}