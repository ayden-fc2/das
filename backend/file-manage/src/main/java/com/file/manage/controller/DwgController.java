package com.file.manage.controller;

import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import com.example.common.service.MyTokenService;
import com.file.manage.service.DwgUploadService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import org.springframework.http.ResponseEntity;

@RestController
@RequestMapping("/dwg")
public class DwgController {
    @Autowired
    MyTokenService ts;

    @Autowired
    DwgUploadService dwgUploadService;

    @Value("${custom.save-file-path-dwg}" + "/")
    private String dwgUploadFilePath;

    // 上传DWG文件并返回文件路径 - controller TODO
    @PostMapping("/upload")
    public ResponseBean<String> uploadDwg(@RequestParam("file") MultipartFile file){
        try {
            String filePath = dwgUploadService.uploadDwg(file);
            return ResponseBean.success(filePath);
        } catch(Exception e){
            throw new MyException(e.getMessage());
        }
    }

    // 下载dwg文件
    @GetMapping("/{fileName}")
    public ResponseEntity<byte[]> downloadFile(@PathVariable String fileName) {
        File file = new File(dwgUploadFilePath + "/" + fileName);
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


