using NumSharp;
using System;
using System.IO;
using System.Linq;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Inception Architecture for Computer Vision
    /// Port from tensorflow\examples\label_image\label_image.py
    /// </summary>
    public class ImageRecognitionInceptionv3 : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Inception v3 Image Recognition";
        public bool IsImportingGraph { get; set; } = false;

        const string root_image_processing = "ImageProcessing";

        string images_folder_for_predicting = Path.Join(root_image_processing, "ImagesForPredictions");

        string inception_v3_assets_dir = Path.Join(root_image_processing, "Inceptionv3Assets");

        string pbFile = "inception_v3_2016_08_28_frozen.pb";
        string labelFile = "imagenet_slim_labels.txt";

        int input_height = 299;
        int input_width = 299;
        int input_mean = 0;
        int input_std = 255;
        string input_name = "import/input";
        string output_name = "import/InceptionV3/Predictions/Reshape_1";

        public bool Run()
        {
            PrepareData();

            var labels = File.ReadAllLines(Path.Join(inception_v3_assets_dir, labelFile));

            //var nd = ReadTensorFromImageFile(Path.Join(dir, picFileToPredict),  //Works for original image

            string picFilePath = Path.Join(images_folder_for_predicting, "PeopleForPrediction", "grace_hopper.jpg");
            //string picFilePath = Path.Join(images_folder_for_predicting, "FlowersForPredictions", "RareThreeSpiralledRose.png");

            var nd = ReadTensorFromImageFile(picFilePath,
                                             input_height: input_height,
                                             input_width: input_width,
                                             input_mean: input_mean,
                                             input_std: input_std);

            var graph = Graph.ImportFromPB(Path.Join(inception_v3_assets_dir, pbFile));
            var input_operation = graph.get_operation_by_name(input_name);
            var output_operation = graph.get_operation_by_name(output_name);

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
                Console.WriteLine($"{picFilePath}: {idx} {labels[(int)idx]}, {results[(int)idx]}");

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
            Directory.CreateDirectory(inception_v3_assets_dir);

            // get model file
            string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";
            
            Utility.Web.Download(url, inception_v3_assets_dir, $"{pbFile}.tar.gz");

            Utility.Compress.ExtractTGZ(Path.Join(inception_v3_assets_dir, $"{pbFile}.tar.gz"), inception_v3_assets_dir);

            // download sample picture
            string pic = "grace_hopper.jpg";
            url = $"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/{pic}";

            string downloadFolder = Path.Join(images_folder_for_predicting, "PeopleForPrediction");
            Utility.Web.Download(url, downloadFolder, pic);
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
