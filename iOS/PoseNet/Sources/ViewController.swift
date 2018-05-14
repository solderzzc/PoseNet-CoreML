import Foundation
import UIKit
import Vision
import TensorSwift

class ViewController: UIViewController {
    
    let model = posenet()
    let ImageWidth = 513
    let ImageHeight = 513
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        runOffline()
        
//        let fname = "tennis_in_crowd.jpg"
//        if let image = UIImage(named: fname){
//            print(measure(runCoreML(image)).duration)
//        }
    }
    func runOffline() {
        
        let scores = getTensor("heatmapScores",[33, 33, 17])
        let offsets = getTensor("offsets",[33, 33, 34])
        let displacementsFwd = getTensor("displacementsFwd",[33, 33, 32])
        let displacementsBwd = getTensor("displacementsBwd",[33, 33, 32])
        
        let poses = decodeMultiplePoses(
            scores: scores,
            offsets: offsets,
            displacementsFwd: displacementsFwd,
            displacementsBwd: displacementsBwd,
            outputStride: 16, maxPoseDetections: 5,
            scoreThreshold: 0.5,nmsRadius: 20)
        
        print(poses)
    }
    
    func runCoreML(_ image: UIImage) {
//        imageView.image = image
        
        let img = image.pixelBuffer(width: ImageWidth, height: ImageWidth)
        let result = try? model.prediction(image__0: img!)

        let names: [String] = ["heatmap__0","offset_2__0","displacement_fwd_2__0","displacement_bwd_2__0"]
        let tensors = names.reduce(into: [String: Tensor]()) {
            $0[$1] = getTensor(
                result?.featureValue(for: $1)?.multiArrayValue)
        }
//        let mlarray = result?.featureValue(for: "heatmap__0")?.multiArrayValue
//
//        let length = mlarray!.count
//        let doublePtr =  mlarray!.dataPointer.bindMemory(to: Double.self, capacity: length)
//        let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: length)
//        let buffer = Array(doubleBuffer)
//        let sum = buffer.reduce(0, +) / (17 * 33 * 33)
//        print(sum)

        let poses = decodeMultiplePoses(
                        scores: tensors["heatmap__0"]!,
                        offsets: tensors["offset_2__0"]!,
                        displacementsFwd: tensors["displacement_fwd_2__0"]!,
                        displacementsBwd: tensors["displacement_bwd_2__0"]!,
                        outputStride: 16, maxPoseDetections: 5,
                        scoreThreshold: 0.5,nmsRadius: 20)
        
        //    let resolution =
        //        getValidResolution(imageScaleFactor, input.width, outputStride);
        
        
        print(poses)
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func measure <T> (_ f: @autoclosure () -> T) -> (result: T, duration: String) {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = f()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        return (result, "Elapsed time is \(timeElapsed) seconds.")
    }
    
    func getTensor(_ mlarray: MLMultiArray!) -> Tensor {
        let length = mlarray.count
        let doublePtr =  mlarray.dataPointer.bindMemory(to: Double.self, capacity: length)
        let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: length)
        let element = Array(doubleBuffer).asArrayOfFloat
        
        let dim = mlarray.shape.map { Dimension(($0 as? Int)!)}
        return Tensor(shape: [dim[0], dim[1], dim[2]], elements: element)
    }
    func getTensor(_ name: String,_ shape: Shape) -> Tensor {
        let url = Bundle.main.url(forResource: name, withExtension: "bin")!
        let binaryData = try! Data(contentsOf: url, options: [])
        let values: [Float32] = binaryData.withUnsafeBytes {
            [Float32](UnsafeBufferPointer(start: $0, count: binaryData.count/MemoryLayout<Float32>.stride))
        }
        return Tensor(shape: shape, elements: values)
    }
}

extension Array where Iterator.Element == Double {
    public var asArrayOfFloat: [Float] {
        return self.map { return Float($0) } // compiler error
    }
}