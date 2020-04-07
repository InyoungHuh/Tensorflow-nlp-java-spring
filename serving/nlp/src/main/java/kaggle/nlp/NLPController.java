package kaggle.nlp;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Controller
public class NLPController {

    private NLPService nlpService;
    @Autowired
    public NLPController(NLPService nlpnService) {
        this.nlpService = nlpnService;
    }

    @RequestMapping(value = "/")
    public String viewHomePage(Model model) {
        return "index";
    }

    @RequestMapping(value = "/save", method = RequestMethod.POST)
    public String sendSentence(String message, Model model) {

        Boolean isReal = nlpService.classify("hello");
        String msg = isReal ? "This is real disaster message" : "This is not real disaster message";
        model.addAttribute("message", msg);

        return "index";
    }


/*
    //@GetMapping(value = "/")
    @RequestMapping("/nlp")
    public String classify(String sentence) {
        Boolean tmp = nlpService.classify(sentence);
        return "nlp";
    }
    */
}
