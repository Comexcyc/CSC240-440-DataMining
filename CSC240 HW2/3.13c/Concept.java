//Yingcan Chen
//3.13c
import java.util.*;

public class Concept{

  class Attribute{
    String name;
    int count;

    Attribute(String name,int count){
      this.name = name;
      this.count = count;
    }
  }

  class Bin{
    String name;
    int max;
    int min;
    int mean;
    int sum;
    int count;
  }

  class categotical{
    //ArrayList<Attribute> count_ary = new ArrayList<>();
    String[] schema = {"Country", "State", "City"};
    Attribute[] count_ary = new Attribute[schema.length];
    String concept_hierarchy[] = new String[schema.length];
    HashMap<String,String> dictionary = new HashMap<>();
    String[] database = {
      "Shanghai","BeiJing","Shenzhen","China"
    };

    categotical(){
      dictionary.put("Shanghai","City");
      dictionary.put("BeiJing","City");
      dictionary.put("Shenzhen","City");
      dictionary.put("China","Country");
    }

    boolean isinAttribute(String name,String Attribute){
      return dictionary.get(name) == Attribute;
    }

    int countatrribute(String[] database, String Attribute){
      HashSet<String> set = new HashSet<>();
      for(String names: database){
        if(isinAttribute(names,Attribute)) set.add(names);
      }
      return set.size();
    }
    void generate(){
      int index = 0;
      for(String A: schema){
        int distinct_count = countatrribute(database,A);
        count_ary[index++] = new Attribute(A,distinct_count);
      }

      Arrays.sort(count_ary, new Comparator<Attribute>(){
        public int compare(Attribute a1, Attribute a2){
          int ans = a1.count >= a2.count ? 1 : -1;
          return ans;
        }
      });

      int index2 = 0;
      for(int i = 0; i < count_ary.length;i++){
        //System.out.println(a.name + "  "+ a.count);
        if(count_ary[i].count != 0) concept_hierarchy[index2++] = count_ary[i].name;
      }
      //System.out.println(count_ary.length);
    }
  }

  class numerical{

  }

  class numerical2{
    Bin concept_hierarchy[];
    String concept_attb;
    int bin_depth;
    int range_min;
    int range_max;

    numerical2(){
      Scanner scanner = new Scanner(System.in);
      System.out.println("Please enter depth of the bin");
      bin_depth = scanner.nextInt();
      System.out.println("Please enter minimum of range");
      range_min = scanner.nextInt();
      System.out.println("Please enter maximum of range");
      range_max = scanner.nextInt();
      concept_hierarchy = new Bin[range_max / bin_depth];
      for(int i = 0;i < concept_hierarchy.length; i++){
        concept_hierarchy[i] = new Bin();
      }
    }

    void generate(){
      for(int i = 0; i < range_max / bin_depth; i++){
        concept_hierarchy[i].name = "level" + i;
        concept_hierarchy[i].min = 0;
        concept_hierarchy[i].max = 0;
      }
      int j = 1;
      int k = 0;


      for(int i = 0; i < concept_hierarchy.length; i++){
        concept_hierarchy[i].mean = concept_hierarchy[i].sum / concept_hierarchy[i].count;
      }
    }
  }
  public static void main(String[] args) {
    Concept con = new Concept();
    categotical cat = con.new categotical();
    cat.generate();


  }
}
