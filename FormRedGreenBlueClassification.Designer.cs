namespace RGBclassifier
{
    partial class FormCrosses
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.pictureBoxCrosses = new System.Windows.Forms.PictureBox();
            this.pictureBoxNeuralNetworkOutput = new System.Windows.Forms.PictureBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.labelEpoch = new System.Windows.Forms.Label();
            this.timerTrainAndPlot = new System.Windows.Forms.Timer(this.components);
            this.pictureBoxOverlayOfOutput = new System.Windows.Forms.PictureBox();
            this.label3 = new System.Windows.Forms.Label();
            this.labelConfidence = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxCrosses)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxNeuralNetworkOutput)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxOverlayOfOutput)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBoxCrosses
            // 
            this.pictureBoxCrosses.Location = new System.Drawing.Point(7, 25);
            this.pictureBoxCrosses.Name = "pictureBoxCrosses";
            this.pictureBoxCrosses.Size = new System.Drawing.Size(258, 214);
            this.pictureBoxCrosses.TabIndex = 0;
            this.pictureBoxCrosses.TabStop = false;
            // 
            // pictureBoxNeuralNetworkOutput
            // 
            this.pictureBoxNeuralNetworkOutput.Location = new System.Drawing.Point(277, 25);
            this.pictureBoxNeuralNetworkOutput.Name = "pictureBoxNeuralNetworkOutput";
            this.pictureBoxNeuralNetworkOutput.Size = new System.Drawing.Size(258, 214);
            this.pictureBoxNeuralNetworkOutput.TabIndex = 1;
            this.pictureBoxNeuralNetworkOutput.TabStop = false;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(7, 6);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(54, 15);
            this.label1.TabIndex = 2;
            this.label1.Text = "Test Data";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(277, 6);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(131, 15);
            this.label2.TabIndex = 3;
            this.label2.Text = "Neural Network Output";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // labelEpoch
            // 
            this.labelEpoch.Location = new System.Drawing.Point(710, 5);
            this.labelEpoch.Name = "labelEpoch";
            this.labelEpoch.Size = new System.Drawing.Size(92, 16);
            this.labelEpoch.TabIndex = 4;
            this.labelEpoch.Text = "Epoch";
            this.labelEpoch.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // timerTrainAndPlot
            // 
            this.timerTrainAndPlot.Enabled = true;
            this.timerTrainAndPlot.Interval = 2;
            // 
            // pictureBoxOverlayOfOutput
            // 
            this.pictureBoxOverlayOfOutput.Location = new System.Drawing.Point(547, 25);
            this.pictureBoxOverlayOfOutput.Name = "pictureBoxOverlayOfOutput";
            this.pictureBoxOverlayOfOutput.Size = new System.Drawing.Size(258, 214);
            this.pictureBoxOverlayOfOutput.TabIndex = 5;
            this.pictureBoxOverlayOfOutput.TabStop = false;
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(547, 5);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(98, 16);
            this.label3.TabIndex = 6;
            this.label3.Text = "Min. confidence:";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // labelConfidence
            // 
            this.labelConfidence.Location = new System.Drawing.Point(644, 6);
            this.labelConfidence.Name = "labelConfidence";
            this.labelConfidence.Size = new System.Drawing.Size(60, 16);
            this.labelConfidence.TabIndex = 7;
            this.labelConfidence.Text = "0%";
            this.labelConfidence.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // FormCrosses
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(814, 246);
            this.Controls.Add(this.labelConfidence);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.pictureBoxOverlayOfOutput);
            this.Controls.Add(this.labelEpoch);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.pictureBoxNeuralNetworkOutput);
            this.Controls.Add(this.pictureBoxCrosses);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.KeyPreview = true;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormCrosses";
            this.ShowIcon = false;
            this.Text = "AI Overfitting Red Green Blue Cross Classifier";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.FormCrosses_KeyDown);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxCrosses)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxNeuralNetworkOutput)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxOverlayOfOutput)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private PictureBox pictureBoxCrosses;
        private PictureBox pictureBoxNeuralNetworkOutput;
        private Label label1;
        private Label label2;
        private Label labelEpoch;
        private System.Windows.Forms.Timer timerTrainAndPlot;
        private PictureBox pictureBoxOverlayOfOutput;
        private Label label3;
        private Label labelConfidence;
    }
}