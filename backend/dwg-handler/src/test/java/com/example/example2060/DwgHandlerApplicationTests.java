package com.example.example2060;

import com.dwg.handler.DwgHandlerApplication;
import com.dwg.handler.dao.InsertStMapper;
import com.dwg.handler.utils.JsonProcessor;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest(classes = DwgHandlerApplication.class)
class DwgHandlerApplicationTests {

    @Value("${custom.save-file-path-dwg}" + "/")
    private String dwgUploadFilePath;

    private JsonProcessor jsonProcessor = new JsonProcessor();

    @Autowired
    InsertStMapper insertStMapper;

    @Test
    void contextLoads() {
        // 测试处理JSON文件
        String inputFile = "../" + dwgUploadFilePath + "/64cf67ca-1bec-441f-a20f-795b61dcbcad.json";
        String outputFile = "../" + dwgUploadFilePath + "/test.json";
        jsonProcessor.processJsonFile(inputFile, outputFile);
    }

}
