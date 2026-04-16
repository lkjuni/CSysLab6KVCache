#include "runState.hpp"
#include <iostream>


CRunState::CRunState()
    : currentActivation(nullptr), branchActivation(nullptr), extraBuffer(nullptr),
      hiddenBuffer(nullptr), extraHiddenBuffer(nullptr), q(nullptr), k(nullptr),
      v(nullptr), attentionScores(nullptr), logits(nullptr),
      keyCache(nullptr), valueCache(nullptr) {}


CRunState::~CRunState() {
}


void CRunState::allocateMemory(CModelConfig* config, bool openKvCache) {
    
    int tDim = openKvCache ? config->dim : (config->maxSeqLen * config->dim);
    int tHiddenDim = openKvCache ? config->feedForwardDim : (config->maxSeqLen * config->feedForwardDim);
    
    
    
    int kvDim = (config->dim * config->numKvHeads) / config->numKvHeads;

    currentActivation = new float[tDim]();
    branchActivation = new float[tDim]();
    extraBuffer = new float[tDim]();
    hiddenBuffer = new float[tHiddenDim]();
    extraHiddenBuffer = new float[tHiddenDim]();
    q = new float[tDim]();
    keyCache = new float[config->numLayers * config->maxSeqLen * kvDim]();
    valueCache = new float[config->numLayers * config->maxSeqLen * kvDim]();
    attentionScores = new float[config->numHeads * config->maxSeqLen]();
    logits = new float[config->vocabSize]();  

    if (!currentActivation || !branchActivation || !extraBuffer || !hiddenBuffer || !extraHiddenBuffer ||
        !q || !keyCache || !valueCache || !attentionScores || !logits) {
        std::cerr << "[ERROR:] Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (openKvCache) {
        keyCache = new float[config->numLayers * config->maxSeqLen * kvDim]();
        valueCache = new float[config->numLayers * config->maxSeqLen * kvDim]();
        attentionScores = new float[config->numHeads * config->maxSeqLen]();

        if (!currentActivation || !branchActivation || !extraBuffer || !hiddenBuffer || !extraHiddenBuffer ||
            !q || !keyCache || !valueCache || !attentionScores || !logits) {
            std::cerr << "[ERROR:] Memory allocation failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        k = new float[config->maxSeqLen * config->dim]();
        v = new float[config->maxSeqLen * config->dim]();
        
        attentionScores = new float[config->maxSeqLen * config->numHeads * config->maxSeqLen]();
        // ensure all mallocs went fine
        if (!currentActivation || !branchActivation || !extraBuffer || !hiddenBuffer || !extraHiddenBuffer ||
            !q || !k || !v
            || !attentionScores || !logits) {
            fprintf(stderr, "malloc failed!\n");
            exit(EXIT_FAILURE);
        }
    }




}

void CRunState::deallocateMemory(bool openKvCache) {
    delete[] currentActivation;
    delete[] branchActivation;
    delete[] extraBuffer;
    delete[] hiddenBuffer;
    delete[] extraHiddenBuffer;
    delete[] q;
    delete[] attentionScores;
    delete[] logits; 
    delete[] keyCache;
    delete[] valueCache;
}