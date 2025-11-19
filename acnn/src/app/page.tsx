"use client"
import { Button } from "acnn_v1/components/ui/button";
import Link from "next/link";
import { 
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "acnn_v1/components/ui/card"
import { useState } from "react";

interface Prediction {
  class: string;
  confidence: number;
}

interface WaveFormData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse {
  predictions: Prediction[];
  waveform: WaveFormData;
}

export default function HomePage() {
  const [vizdata, setvizdata] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);
  
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setSelectedFile(file);
    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setvizdata(null);
    
    const reader = new FileReader();
    
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            ""
          )
        );
        
        const resp = await fetch(
          "https://notvivek230588--audio-cnn-inference-audio-classifier-inference.modal.run",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64String })
          }
        );
        
        if (!resp.ok) {
          throw new Error(`API error: ${resp.statusText}`);
        }
        
        const data: ApiResponse = await resp.json();
        setvizdata(data);
        
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown Error!");
      } finally {
        setIsLoading(false);
      }
    };
    
    reader.onerror = () => {
      setError("Failed to read file");
      setIsLoading(false);
    };
    
    reader.readAsArrayBuffer(file);
  };

  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[60%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900">
            ACNN Classifier Demo (Resnet Arch)
          </h1>
          
          <p className="text-md mb-8 text-stone-600">
            Upload the WAV file to see predictions
          </p>
          
          <div className="flex flex-col items-center">
            <div className="relative inline-block">
              <input
                type="file"
                accept=".wav"
                id="file_upload"
                onChange={handleFileChange}
                disabled={isLoading}
                className="absolute inset-0 w-full h-full cursor-pointer opacity-0"
              />
              <Button
                size="lg"
                className="border-stone-300"
                variant="outline"
                disabled={isLoading}
              >
                {isLoading ? "Analysing..." : selectedFile ? selectedFile.name : "Choose File"}
              </Button>
            </div>
            
            {selectedFile && !isLoading && (
              <p className="mt-4 text-sm text-stone-500">
                Selected: {selectedFile.name}
              </p>
            )}
          </div>
        </div>
        
        {error && (
          <Card className="mb-8 border-red-200 bg-red-50">
            <CardContent>
              <p className="pt-6 text-red-600">Error: {error}</p>
            </CardContent>
          </Card>
        )}
        
        {vizdata && (
          <div className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle>TOP PREDICTIONS</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {vizdata.predictions.map((pred, idx) => (
                    <div key={idx} className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium text-stone-900">
                            {pred.class}
                          </span>
                          <span className="text-sm text-stone-600">
                            {(pred.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full bg-stone-200 rounded-full h-2">
                          <div
                            className="bg-stone-900 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${pred.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {vizdata.waveform && (
              <Card>
                <CardHeader>
                  <CardTitle>WAVEFORM DATA</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm text-stone-600">
                    <p>Sample Rate: {vizdata.waveform.sample_rate} Hz</p>
                    <p>Duration: {vizdata.waveform.duration.toFixed(2)} seconds</p>
                    <p>Samples: {vizdata.waveform.values.length}</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </main>
  );
}