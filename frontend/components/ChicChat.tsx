"use client";

import { useState, useRef } from "react";
import { Send, Sparkles, Upload } from "lucide-react"; // Added Upload icon
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import ReactMarkdown from 'react-markdown';

// Function to query the Flask API for text-based search
const queryFashionAPI = async (query: string) => {
  const response = await fetch("http://localhost:5000/query-fashion", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  return response.json();
};

// Function to query the Flask API for image-to-image search
const imageToImageSearch = async (file: File) => {
  const formData = new FormData();
  formData.append('image', file);
  const response = await fetch("http://localhost:5000/image-to-image-search", {
    method: "POST",
    body: formData,
  });
  return response.json();
};

type Message = {
  text: string;
  sender: "user" | "bot";
  image?: string;
};

export default function ChicChat() {
  const [messages, setMessages] = useState<Message[]>([
    { text: "Hello! I'm Chic Chat, your personal fashion assistant. How can I help you style your outfit today? You can also upload an image to find similar styles!", sender: "bot" },
  ]);
  const [input, setInput] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = async () => {
    if (input.trim()) {
      // Add user message
      setMessages((prev) => [...prev, { text: input, sender: "user" }]);

      // Fetch response from Flask API
      const apiResponse = await queryFashionAPI(input);

      // Add bot response and format using react-markdown
      setMessages((prev) => [...prev, { text: apiResponse.response, sender: "bot" }]);

      // Display images from the API response
      apiResponse.images.forEach((image: string) => {
        setMessages((prev) => [...prev, { text: "Here's an outfit suggestion!", sender: "bot", image }]);
      });

      // Clear input field
      setInput("");
    }
  };

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Add the uploaded image message first
      const uploadedImageUrl = URL.createObjectURL(file); // This creates a temporary URL for the uploaded image
      setMessages((prev) => [
        ...prev,
        { text: "Uploaded image", sender: "user", image: uploadedImageUrl },
        { text: "Searching for similar styles...", sender: "bot" }
      ]);
  
      try {
        const apiResponse = await imageToImageSearch(file);
  
        if (apiResponse.images && apiResponse.images.length > 0) {
          // Ignore the first image and only show the last two images
          const filteredImages = apiResponse.images.slice(1, 3); // Slicing to get only the 2nd and 3rd images
  
          setMessages((prev) => [
            ...prev,
            { text: "Here are some similar styles I found:", sender: "bot" }
          ]);
  
          // Add each similar style suggestion image, excluding the first one
          filteredImages.forEach((image: string) => {
            setMessages((prev) => [
              ...prev,
              { text: "Similar style suggestion", sender: "bot", image }
            ]);
          });
        } else {
          setMessages((prev) => [
            ...prev,
            { text: "I couldn't find any similar styles. Try uploading a different image or asking for text-based suggestions.", sender: "bot" }
          ]);
        }
      } catch (error) {
        setMessages((prev) => [
          ...prev,
          { text: "Sorry, there was an error processing your image. Please try again.", sender: "bot" }
        ]);
      }
    }
  };
  
  

  return (
    <div className="flex h-[600px] w-full max-w-md flex-col overflow-hidden rounded-xl bg-gradient-to-br from-pink-300 via-purple-300 to-indigo-400 p-4 shadow-xl">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Chic Chat</h2>
        <Sparkles className="h-6 w-6 text-yellow-300" />
      </div>
      <ScrollArea className="flex-1 rounded-lg bg-white bg-opacity-50 p-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`mb-4 max-w-[80%] rounded-lg p-3 ${
              message.sender === "user" ? "ml-auto bg-indigo-100" : "mr-auto bg-pink-100"
            }`}
          >
            <ReactMarkdown className="text-sm text-gray-800">{message.text}</ReactMarkdown>
            {message.image && (
              <img
                src={message.image}
                alt="Outfit suggestion"
                className="mt-2 w-full max-h-64 rounded-md object-contain"
                onError={(e) => e.currentTarget.style.display = 'none'}
              />
            )}
          </div>
        ))}
      </ScrollArea>
      <div className="mt-4 flex items-center gap-2">
        <Input
          type="text"
          placeholder="Ask for outfit suggestions..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          className="flex-1 rounded-full bg-white bg-opacity-50 px-4 py-2 text-gray-800 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />
        <Button onClick={handleSend} className="rounded-full bg-purple-500 p-2 text-white hover:bg-purple-600">
          <Send className="h-5 w-5" />
        </Button>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          ref={fileInputRef}
          className="hidden"
        />
        <Button onClick={() => fileInputRef.current?.click()} className="rounded-full bg-purple-500 p-2 text-white hover:bg-purple-600">
          <Upload className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}
