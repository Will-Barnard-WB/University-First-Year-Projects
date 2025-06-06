import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

public class EchoServer {

    private int port;

    public EchoServer(int port) {

        this.port = port;
    }


    private void run(){

        try(ServerSocket ss = new ServerSocket(port)){

            // listen for a client, method blocks

            Socket client = ss.accept();

            // create stream objects to send data

            PrintWriter sendTo = new PrintWriter(   client.getOutputStream());

            BufferedReader readFrom = new BufferedReader(new InputStreamReader(client.getInputStream()));

            // accept a message, also blocks
            String inputLine = readFrom.readLine();

            while(inputLine != null){

                // send the message back
                sendTo.println(inputLine);
                sendTo.flush();
                inputLine = readFrom.readLine();

            }
        }
        catch(IOException ioe){
            ioe.printStackTrace();
        }

    }

    public static void main(String[] args){

        EchoServer server = new EchoServer(7777);
        server.run();

    }

}