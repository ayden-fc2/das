package com.example.example2060;

import com.dwg.handler.DwgHandlerApplication;
import com.dwg.handler.service.impl.JsonProcessor;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@SpringBootTest(classes = DwgHandlerApplication.class)
class DwgHandlerApplicationTests {

    @Value("${custom.save-file-path-dwg}" + "/")
    private String dwgUploadFilePath;

    private JsonProcessor jsonProcessor = new JsonProcessor();

    @Test
    void contextLoads() {
        // 测试处理JSON文件
        String inputFile = "../" + dwgUploadFilePath + "/820c80c1-b5b2-46ca-95b7-5ec2267cb2dc.json";
        String outputFile = "../" + dwgUploadFilePath + "/820c80c1-b5b2-46ca-95b7-5ec2267cb2dc_handled.json";
        jsonProcessor.processJsonFile(inputFile, outputFile);
    }

}
