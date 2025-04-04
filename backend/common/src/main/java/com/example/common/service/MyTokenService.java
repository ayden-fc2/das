package com.example.common.service;

import java.util.List;
import java.util.Map;

public interface MyTokenService {
    int tokenToUserId(String token);

    Map<String, List<String>> tokenToClaims(String token);

}
