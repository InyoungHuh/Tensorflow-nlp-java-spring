package kaggle.nlp;

import org.springframework.beans.factory.annotation.Autowired;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

@Service
public class NLPService {
    private final Session modelBundleSession;
    private static Lemmatisation lemma;

    private final static String FEED_OPERATION = "serving_default_embedding_input";
    private final static String FETCH_OPERATION_CLASS_ID = "StatefulPartitionedCall";

    @Autowired
    public NLPService() {
        this.modelBundleSession = SavedModelBundle.load("src/main/resources/model/disaster", "serve").session();
        this.lemma = new Lemmatisation();
    }

    private static HashMap<String, Integer> loadBagOfWords() {
        HashMap<String, Integer> bow = new HashMap<>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader("src/main/resources/model/bow.txt"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        try {
            String line = br.readLine();
            while (line != null) {
                String[] strArr = line.split(" ");
                bow.put(strArr[0], Integer.parseInt(strArr[1]));
                line = br.readLine();
            }
        } catch (IOException e) {
            System.out.println("cannot load bow file");
        }
        return bow;
    }
    private static Tensor createInputTensor(String sentence) {
        // (taken from the saved_model, node dnn/input_from_feature_columns/input_layer/concat)
        List<String> lemmaStr = lemma.lemmatize(sentence);
        HashMap<String, Integer> bow = loadBagOfWords();

        //inputLen depends on model
        int inputLen = 23;
        float[] input = new float[23];
        
        List<Integer> wordToIdx = new ArrayList<>();
        for(String lemma : lemmaStr)
            if(bow.containsKey(lemma)) wordToIdx.add(bow.get(lemma));

        for(int i = inputLen - wordToIdx.size(), j=0; i< inputLen; i++, j++)
            input[i] = wordToIdx.get(j);

        float[][] data = new float[1][input.length];
        data[0] = input;
        return Tensor.create(data);
    }

    public Boolean classify(String sentence){
        Tensor inputTensor = NLPService.createInputTensor(sentence);

        List<Tensor<?>> result = this.modelBundleSession.runner()
                .feed(NLPService.FEED_OPERATION, inputTensor)
                .fetch(NLPService.FETCH_OPERATION_CLASS_ID)
                .run();

        float[][] value = result.get(0).copyTo(new float[1][1]);

        return value[0][0] > 0.5 ? true : false;
    }


}
