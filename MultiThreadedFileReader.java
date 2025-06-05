import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class PopThread implements Runnable {

    private static int fileCount = 1;
    private static int maxFiles = 0;
    private static boolean newFile = true;

    private ArrayList<String> files;
    private ArrayList<Integer> positions = new ArrayList<>();
    private ArrayList<String> contents = new ArrayList<>();

    @Override
    public void run() {
        while (fileCount <= maxFiles) {
            if (positions.contains(fileCount)) {

                int pos = positions.indexOf(fileCount);
                String content = contents.get(pos);

                writeToFile(content);
                incrementFileCount();
            }
        }
    }

    public PopThread(ArrayList<String> fileNames) {
        files = fileNames;

        for (String fileName : files) {
            String text = (processFile(fileName));
            contents.add(text);
        }
    }

    private static synchronized void incrementFileCount(){
        fileCount ++;
    }

    private static synchronized  void changeMaxFiles(int newMaxFiles){
        maxFiles = newMaxFiles;
    }

    private static synchronized  void changeNewFile(){
        newFile = !newFile;
    }

    private static synchronized void writeToFile(String textToWrite){
        try {
            BufferedWriter bw = null;
            try{
                if (newFile){
                    bw = new BufferedWriter(new FileWriter("result.txt"));
                    changeNewFile();
                }
                else {
                    bw = new BufferedWriter(new FileWriter("result.txt", true));
                }
                bw.write(textToWrite);
            } catch (IOException e) {
                return;
            } finally {
                if(bw != null) {
                    bw.close();
                }
            }
        } catch(IOException e) {
            return;
        }
    }

    private String processFile(String filename) {
        try {
            String text = "";
            String line;
            BufferedReader br = null;

            try {
                br = new BufferedReader(new FileReader(filename));
                line = br.readLine();
                text = line + "\n";

                while (line != null) {
                    line = br.readLine();

                    if (line != null) {

                        if (line.matches("#\\d+/\\d+")) { 
                            
                            String[] Nums = line.substring(1).split("/");
                            int numPos = Integer.parseInt(Nums[0]);
                            int maxPos = Integer.parseInt(Nums[1]);

                            if (maxPos > maxFiles) {
                                changeMaxFiles(maxPos);
                            }

                            positions.add(numPos);

                        }

                        text = text + line + "\n";
                    }
                }
            } catch (FileNotFoundException e) {
                return null;

            } catch (IOException e) {
                return null;

            } finally {
                if (br != null) {
                    br.close();
                }
            }
            return text;

        } catch (IOException e) {
            return null;
        }
    }

}


