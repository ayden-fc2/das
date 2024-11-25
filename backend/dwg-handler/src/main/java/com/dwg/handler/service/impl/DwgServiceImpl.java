package com.dwg.handler.service.impl;

import com.dwg.handler.dao.UploadDwgStMapper;
import com.dwg.handler.entity.UploadDwgSt;
import com.dwg.handler.service.DwgService;
import com.example.common.exception.MyException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Date;

@Service
public class DwgServiceImpl implements DwgService {

    @Value("${custom.save-file-path-dwg}" + "/")
    private String dwgUploadFilePath;

    @Value("${custom.back-file-path-dwg}" + "/")
    private String dwgReturnUrl;

    @Autowired
    UploadDwgStMapper uploadDwgStMapper;

    @Override
    public Boolean genAnalysis(int userId, String projectName, String dwgPath, int isPublic) {
        String file;
        if (dwgPath.startsWith(dwgReturnUrl)) {
            file = dwgPath.substring(dwgReturnUrl.length());
        } else {
            throw new MyException("The dwgPath does not start with dwgReturnUrl");
        }
        String fileName = file.split("\\.")[0];
        String extension = file.split("\\.")[1];
        if (!extension.equals("dwg")) {
            throw new MyException("The extension is not dwg");
        }
        String fullPath = dwgUploadFilePath + file;

        // 需要执行的命令
        String command = "dwgread " + fullPath + " -O JSON -o " + dwgUploadFilePath + fileName + ".json";
        System.out.println(command);
        // 等待命令执行完毕
        try {
            // 使用 Runtime 执行命令
            Process process = Runtime.getRuntime().exec(command);
            // 获取命令的输出
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
            int exitCode = process.waitFor();
            System.out.println("Command executed with exit code: " + exitCode);
            String jsonPath = dwgReturnUrl + fileName + ".json";
            // 新增一条数据
            UploadDwgSt newDwgAna = new UploadDwgSt();
            newDwgAna.setProjectName(projectName);
            newDwgAna.setDwgPath(dwgPath);
            newDwgAna.setIsPublic(isPublic);
            newDwgAna.setJsonPath(jsonPath);
            newDwgAna.setUserId(userId);
            return uploadDwgStMapper.insert(newDwgAna);
        } catch (Exception e) {
            e.printStackTrace();
            throw new MyException(e.getMessage());
        }
    }
}
