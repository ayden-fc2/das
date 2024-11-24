package com.file.manage.controller;

import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import com.file.manage.service.DwgUploadService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

@RestController
@RequestMapping("/common")
public class CommonFileController {

    @Autowired
    DwgUploadService dwgUploadService;

    @Value("${custom.save-file-path-common}" + "/")
    private String commonUploadFilePath;

    //所有人测试
    @RequestMapping("/test")
    public ResponseBean allTestFuc(){return ResponseBean.success("hello");}

    // 上传普通文件并返回文件路径
    @PostMapping("/upload")
    public ResponseBean<String> uploadDwg(@RequestParam("file") MultipartFile file){
        try {
            String filePath = dwgUploadService.uploadFile(file);
            return ResponseBean.success(filePath);
        } catch(Exception e){
            throw new MyException(e.getMessage());
        }
    }

    // 下载文件
    @GetMapping("/{fileType}/{fileName}")
    public ResponseEntity<byte[]> downloadFile(@PathVariable String fileType, @PathVariable String fileName) {
        File file = new File(commonUploadFilePath + "/" + fileType + "/" + fileName);
        if (!file.exists()) {
            throw new MyException("Fill does not exist");
        }
        try {
            byte[] content = Files.readAllBytes(file.toPath());
            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + fileName + "\"")
                    .body(content);
        } catch (IOException e) {
            throw new MyException(e.getMessage());
        }
    }
}
