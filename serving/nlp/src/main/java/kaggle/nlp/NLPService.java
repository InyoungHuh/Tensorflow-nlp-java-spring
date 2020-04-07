package kaggle.nlp;

import org.springframework.beans.factory.annotation.Autowired;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class NLPService {
    private final Session modelBundleSession;

    private final static String FEED_OPERATION = "serving_default_embedding_input";
    private final static String FETCH_OPERATION_CLASS_ID = "StatefulPartitionedCall";

    @Autowired
    public NLPService() {
        this.modelBundleSession = SavedModelBundle.load("src/main/resources/model/disaster", "serve").session();
    }
    private static Tensor createInputTensor(String sentence){
        // order of the data on the input: PetalLength, PetalWidth, SepalLength, SepalWidth
        // (taken from the saved_model, node dnn/input_from_feature_columns/input_layer/concat)
        float[] input = {     0.0f,0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,
                0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,  253.0f, 1665.0f, 7976.0f, 4935.0f};
        float[][] data = new float[1][input.length];
        data[0] = input;
        return Tensor.create(data);
    }

    public Boolean classify(String sentence) {
        Tensor inputTensor = NLPService.createInputTensor(sentence);

        List<Tensor<?>> result = this.modelBundleSession.runner()
                .feed(NLPService.FEED_OPERATION, inputTensor)
                .fetch(NLPService.FETCH_OPERATION_CLASS_ID)
                .run();

        float[][] value = result.get(0).copyTo(new float[1][1]);

        return value[0][0] > 0.5 ? true : false;
    }


}
