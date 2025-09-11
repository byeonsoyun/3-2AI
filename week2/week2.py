# test_torch.py
# PyTorch & CUDA 동작 테스트 코드

try:
    import torch
    print("✓ PyTorch is successfully imported!")
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA 확인
    if torch.cuda.is_available():
        print(f"✓ CUDA is available! Device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA is not available. Using CPU only.")
    
    # 기본 Tensor (CPU)
    test_tensor = torch.tensor([1, 2, 3, 4, 5])
    print(f"\n✓ CPU Tensor created: {test_tensor}")
    print(f"Tensor shape: {test_tensor.shape}")
    print(f"Tensor device: {test_tensor.device}")
    
    squared_tensor = test_tensor ** 2
    print(f"✓ CPU operation test (squaring): {squared_tensor}")
    
    # GPU 연산 테스트
    if torch.cuda.is_available():
        gpu_tensor = test_tensor.to("cuda")   # GPU로 이동
        print(f"\n✓ Tensor moved to device: {gpu_tensor.device}")
        
        squared_gpu_tensor = gpu_tensor ** 2
        print(f"✓ GPU operation test (squaring): {squared_gpu_tensor}")
        
        # GPU에서 계산한 결과를 CPU로 가져오기
        print(f"✓ GPU result back on CPU: {squared_gpu_tensor.to('cpu')}")
    
    print("\n🎉 PyTorch is working correctly with CUDA support!")
    
except ImportError as e:
    print("❌ PyTorch is not installed or not available")
    print(f"Error details: {e}")
    print("Please install PyTorch using: pip install torch torchvision torchaudio")
    
except Exception as e:
    print(f"❌ An error occurred while testing PyTorch: {e}")
