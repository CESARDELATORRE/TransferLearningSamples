using NumSharp;
using System;
using System.IO;
using System.Linq;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    public class ImageRecognitionCustomModelFromAzureCustomVisionOnMLNET : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Image Recognition Custom Model trained/exported from Azure CS Custom Vision, running on ML.NET";
        public bool IsImportingGraph { get; set; } = false;
        const string root_image_processing = "ImageProcessing"; //Root folder name

        // CHAIRS PREDICTION SCENARIO
        //
        string image_filename_to_use_for_prediction = "high-metal-office-chair.jpg";  //north_west_us_wooden_chair.png //high-metal-office-chair.jpg  //green-office-chair-test.jpg 
        string images_folder_for_predicting = Path.Join(root_image_processing, 
                                                        "ImagesForPredictions",
                                                        "ChairsForPredictions"
                                                       );

        string custom_model_assets_dir = Path.Join(root_image_processing,
                                                   "CustomTensorFlowModels"
                                                  );

        string pbFile = "azure_cs_custom_vision_chair_image_classification_exported_model.pb";
        string labelFile = "azure_cs_custom_vision_chair_image_classification_exported_labels.txt";
        //
        ////////////////////////////////////////////////////////////////////
        
        // IMAGE SETTINGS
        int input_height = 227;
        int input_width = 227;
        int input_mean = 0;
        int input_std = 255;   //----> ???????????????????

        // INPUT/OUTPUT TENSORS SETTINGS
        string input_name = "Placeholder"; 
        string output_name = "loss"; 

        public bool Run()
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            throw new NotImplementedException();
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
