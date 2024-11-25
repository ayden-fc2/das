package com.dwg.handler.service.impl;

import com.dwg.handler.dao.UploadDwgStMapper;
import com.dwg.handler.entity.UploadDwgSt;
import com.dwg.handler.service.ReadService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;

@Service
public class ReadServiceImpl implements ReadService {

    @Autowired
    UploadDwgStMapper uploadDwgStMapper;

    @Override
    public List<UploadDwgSt> getPublicList() {
        return uploadDwgStMapper.getPublicList();
    }
}
