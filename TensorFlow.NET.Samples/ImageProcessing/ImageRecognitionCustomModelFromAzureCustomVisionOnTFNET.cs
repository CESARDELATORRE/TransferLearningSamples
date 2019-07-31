using NumSharp;
using System;
using System.IO;
using System.Linq;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    public class ImageRecognitionCustomModelFromAzureCustomVisionOnTFNET : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Image Recognition Custom Model trained/exported from Azure CS Custom Vision, running on TensorFlow.NET";
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
        int input_mean = 117;
        int input_std = 255;   //----> ???????????????????

        // INPUT/OUTPUT TENSORS SETTINGS
        string input_name = "Placeholder"; 
        string output_name = "loss"; 

        public bool Run()
        {
            //No needed for Custom model since .pb model is already local
            //PrepareData();

            var labels = File.ReadAllLines(Path.Join(custom_model_assets_dir, labelFile));

            string picFilePath = Path.Join(images_folder_for_predicting, image_filename_to_use_for_prediction);
            
            var nd = ReadTensorFromImageFile(picFilePath,
                                             input_height: input_height,
                                             input_width: input_width,
                                             input_mean: input_mean,
                                             input_std: input_std);

            var graph = Graph.ImportFromPB(Path.Join(custom_model_assets_dir, pbFile), "");



            Tensor input_operation = graph.OperationByName(input_name);
            Tensor output_operation = graph.OperationByName(output_name);

            // OLD with Inceptionv3
            //var input_operation = graph.get_operation_by_name(input_name);
            //var output_operation = graph.get_operation_by_name(output_name);

            var results = with(tf.Session(graph),
                sess => sess.run(output_operation.outputs[0],
                    new FeedItem(input_operation.outputs[0], nd)));

            results = np.squeeze(results);

            var argsort = results.argsort<float>();
            var top_k = argsort.Data<float>()
                .Skip(results.size - 5)
                .Reverse()
                .ToArray();

            foreach (float idx in top_k)
                Console.WriteLine($"{image_filename_to_use_for_prediction}: Label-Index:{idx} Probability for Label: {labels[(int)idx]} is {results[(int)idx]}");

            return true;
        }

        private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 299,
                                int input_width = 299,
                                int input_mean = 0,
                                int input_std = 255)
        {
            return with(tf.Graph().as_default(), graph =>
            {
                var file_reader = tf.read_file(file_name, "file_reader");
                var image_reader = tf.image.decode_jpeg(file_reader, channels: 3, name: "jpeg_reader");
                var caster = tf.cast(image_reader, tf.float32);
                var dims_expander = tf.expand_dims(caster, 0);
                var resize = tf.constant(new int[] { input_height, input_width });
                var bilinear = tf.image.resize_bilinear(dims_expander, resize);
                var sub = tf.subtract(bilinear, new float[] { input_mean });
                var normalized = tf.divide(sub, new float[] { input_std });

                return with(tf.Session(graph), sess => sess.run(normalized));
            });
        }

        public void PrepareData()
        {
            // OLD for Inception v3
            //
            //Directory.CreateDirectory(custom_model_assets_dir);

            //// get model file
            //string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";
            
            //Utility.Web.Download(url, custom_model_assets_dir, $"{pbFile}.tar.gz");

            //Utility.Compress.ExtractTGZ(Path.Join(custom_model_assets_dir, $"{pbFile}.tar.gz"), custom_model_assets_dir);

            //// download sample picture
            //string pic = "grace_hopper.jpg";
            //url = $"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/{pic}";

            //string downloadFolder = Path.Join(images_folder_for_predicting, "PeopleForPrediction");
            //Utility.Web.Download(url, downloadFolder, pic);
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
