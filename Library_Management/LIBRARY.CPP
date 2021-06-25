#include<iostream.h>
#include<iomanip.h>
#include<fstream.h>
#include<process.h>
#include<conio.h>
#include<string.h>
#include<stdio.h>
  int auto_book()
   {
	int c=0,d=0;
	fstream file;
	file.open("SNO.dat",ios::binary|ios::in|ios::out|ios::nocreate);
	if(file)
	 {
	  file.seekp(-2,ios::end);
	  file.read((char*)&c,sizeof(c));
	  d=++c;
	  file.write((char*)&d,sizeof(d));
	  }
	else
	{
	 file.open("SNO.dat",ios::binary|ios::out);
	 c=1;
	 file.write((char*)&c,sizeof(c));

	}
	return c;
   }
int auto_member()
{ clrscr();
  int c=0;
  int d=0;
  fstream file;
  file.open("SN.dat",ios::binary|ios::in|ios::out|ios::nocreate);
  if(file)
  { file.seekp(-2,ios::end);
    file.read((char*)&c,sizeof(c));
    d=++c;
    file.write((char*)&d,sizeof(d));
  }
  else
  {
   file.open("SN.dat",ios::binary|ios::out);
   c=1;
   file.write((char*)&c,sizeof(c));
  }
  return c;
}

class book
{	int bno;
	char bname[20],author[20],publisher[20];
      public:
      int tn,issued;
	void read_book()
	{       bno=auto_book();
		cout<<"\nEnter book name ";
		gets(bname);
		cout<<"\nEnter author name ";
		gets(author);
		cout<<"\nEnter publisher name ";
		gets(publisher);
		cout<<"\nEnter no. of copies ";
		cin>>tn;
		issued=0;

	}
	void show_book()
	{
		cout<<endl<<bno<<setw(5)<<"\t"<<bname<<setw(15)<<author<<setw(15)<<publisher<<setw(15)<<tn<<setw(5)<<issued;
	}
	void book_modify()
	{ 				char b[20];
					char a[20];
					char p[20];
					cout<<"\nEnter new details ";
					cout<<"\nEnter new book name (Enter $ to retain old one) ";
					gets(b);
					cout<<"\nEnter new author name (Enter # to retain old one)";
					gets(a);
					cout<<"\nEnter new publicatons (Enter @ to retain old one)";
					gets(p);
					if(strcmpi(b,"$")!=0)
						strcpy(bname,b);
					if(strcmpi(a,"#")!=0)
						strcpy(author,a);
					if(strcmpi(p,"@")!=0)
						strcpy(publisher,p);
	}
	int retbno()
	{	return bno;
	}
	int rettot()
	{	return tn;
	}
	int retiss()
	{	return issued;
	}
};
class member
{
	int mno;
	char mname[15],address[20],phone[15],gender;
  public:
	int book;
	void read_member()
	{      	mno = auto_member();
		cout<<"\nEnter member name ";
		gets(mname);
		cout<<"\nEnter member address ";
		gets(address);
		cout<<"\nEnter phone no. ";
		gets(phone);
		cout<<"\nGender: M / F \n";
		cin>>gender;
		book=0;
		cout<<"\nMEMBER RECORD  CREATED";
	}
	void print_member()
	{	cout<<endl<<mno<<"\t"<<setw(7)<<mname<<setw(15)<<address<<setw(15)<<phone<<"\t\t"<<gender<<setw(10)<<book;
	}
	void member_modify()
	{			      	char n[20];
					char a[20];
					char p[20];
					char g;
					cout<<"\nEnter new details ";
					cout<<"\nEnter new member name (Enter $ to retain old one) ";
					gets(n);
					cout<<"\nEnter new address (Enter # to retain old one)";
					gets(a);
					cout<<"\nEnter new phone no. (Enter @ to retain old one)";
					gets(p);
					cout<<"\nGENDER: M / F ";
					cin>>g;
					gender = g;
					if(strcmpi(n,"$")!=0)
						strcpy(mname,n);
					if(strcmpi(a,"#")!=0)
						strcpy(address,a);
					if(strcmpi(p,"@")!=0)
						strcpy(phone,p);
	}
	int retmno()
	{	return mno;
	}
	int retbno()
	{
		return book;
	}
};
void write_book()
{       clrscr();
	char ch;
	fstream ofile;
	ofile.open("book.dat",ios::binary|ios::app);
	book b;
	do
	{	b.read_book();
		ofile.write((char*)&b,sizeof(b));
		cout<<"\nDo u want to continue ";
		cin>>ch;
	}while((ch=='y')||(ch=='Y'));
	//ofile.close();
}
void write_member()
{       clrscr();
	char ch;
	fstream ofile;
	ofile.open("member.dat",ios::binary|ios::app);
	member m;
	do
	{	m.read_member();
		ofile.write((char*)&m,sizeof(m));
		cout<<"\nDo u want to continue ";
		cin>>ch;
	}while((ch=='y')||(ch=='Y'));
	//ofile.close();
}
void del()
{       clrscr();
	int ch;
	do
	{
	cout<<"\n\n DELETE MENU ";
	cout<<"\n1.BOOK ";
	cout<<"\n2.MEMBER ";
	cout<<"\n3.BACK";
	cout<<"\nEnter your choice ";
	cin>>ch;
	switch(ch)
	{
		case 1: clrscr();
			int bno;
			ifstream bifile1;
			book b;
			bifile1.open("book.dat",ios::binary);
			cout<<"\nBNO"<<setw(7)<<"\t"<<"BNAME"<<setw(15)<<"AUTHOR"<<setw(15)<<"PUBLICATION"<<setw(15)<<"TOTAL"<<setw(5)<<"\tISSUED";
			while(bifile1.read((char*)&b,sizeof(b)))
			{
				b.show_book();
			}
			bifile1.close();
			cout<<"\n\n\nEnter book no. to be deleted ";
			cin>>bno;
			ifstream bifile;
			bifile.open("book.dat",ios::binary);
			ofstream bofile;
			bofile.open("temp.dat",ios::binary);
			while(bifile.read((char*)&b,sizeof(b)))
			{
				if(b.retbno()!=bno)
				bofile.write((char*)&b,sizeof(b));
			}
			bifile.close();
			bofile.close();
			remove("book.dat");
			rename("temp.dat","book.dat");
			ifstream bifil;
			bifil.open("book.dat",ios::binary);
			cout<<"\n\t\t\tAFTER DELETION ";
			cout<<"\nBNO"<<setw(7)<<"\t"<<"BNAME"<<setw(15)<<"AUTHOR"<<setw(15)<<"PUBLICATION"<<setw(15)<<"TOTAL"<<setw(5)<<"\tISSUED";
			while(bifil.read((char*)&b,sizeof(b)))
			{
				b.show_book();
			}
			bifil.close();
			break;
		case 2: clrscr();
			int mno;
			ifstream mifile1;
			member m;
			mifile1.open("member.dat",ios::binary);
			cout<<"\nMNO."<<setw(7)<<"\t"<<"MNAME"<<setw(15)<<"ADDRESS"<<setw(15)<<"PHONE"<<setw(15)<<"GENDER"<<setw(7)<<"\tBOOK NO.";
			while(mifile1.read((char*)&m,sizeof(m)))
			{	m.print_member();
			}
			mifile1.close();
			cout<<"\n\n\nEnter member no. to be deleted ";
			cin>>mno;
			ifstream mifile;
			mifile.open("member.dat",ios::binary);
			ofstream mofile;
			mofile.open("temp.dat",ios::binary);
			while(mifile.read((char*)&m,sizeof(m)))
			{
				if(m.retmno()!=mno)
					mofile.write((char*)&m,sizeof(m));
			}
			mifile.close();
			mofile.close();
			remove("member.dat");
			rename("temp.dat","member.dat");
			ifstream mifil;
			mifil.open("member.dat",ios::binary);
			cout<<"\n\t\t\tAFTER DELETION ";
			cout<<"\nMNO."<<setw(7)<<"\t\t"<<"MNAME"<<setw(15)<<"ADDRESS"<<setw(15)<<"PHONE"<<setw(15)<<"GENDER"<<setw(7)<<"\tBOOK NO.";
			while(mifil.read((char*)&m,sizeof(m)))
			{	m.print_member();
			}
			mifil.close();
			break;
		case 3: break;
	}
	}while(ch!=3);

}
void issue_book()
{       clrscr();
	fstream ifile,ifile1;
	int mn,bn,flag1=0,flag2=0,flag3=0,check;
	long pos,pos1;
	member m;
	cout<<"\nEnter member no. ";
	cin>>mn;
	ifile.open("member.dat",ios::binary|ios::in|ios::out);
	while(!ifile.eof())
	{       pos1=ifile.tellg();
		ifile.read((char*)&m,sizeof(m));
		if(m.retmno()==mn)
		{     	flag1=1;
			break;
		}
	}
	if(flag1==1)
	{	cout<<"\nEnter book no. ";
		cin>>bn;
		book b;
		ifile1.open("book.dat",ios::binary|ios::in|ios::out);
		while(!ifile1.eof())
		{	pos=ifile1.tellg();
			ifile1.read((char*)&b,sizeof(b));
			if(m.book==0)
			{       check=1;
				if(b.retbno()==bn)
				{       flag2=1;
					int bt = b.rettot();
					int bi = b.retiss();
					if(bi<bt)
					{       flag3=1;
						bi = bi+1;
						m.book = bn;
						b.issued=bi;
						ifile1.seekg(pos);
						ifile1.write((char*)&b,sizeof(b));
						ifile.seekg(pos1);
						ifile.write((char*)&m,sizeof(m));
					}
				}
			}
		}
	}
	else
		cout<<"\nMember not available ";
	if(flag2==0)
		cout<<"\nBook not available";
	if(flag3==1)
		cout<<"\nBook issued ";
	else
		cout<<"\nCopies not available ";
	if(check==0)
		cout<<"\nReturn the book u last took";
}
void return_book()
{       clrscr();
	fstream ifile,ifile1;
	int mn,bn,flag1=0,flag2=0,flag3=0;
	long pos,pos1;
	cout<<"\nEnter member no. ";
	cin>>mn;
	ifile.open("member.dat",ios::binary|ios::in|ios::out);
	member m;
	book b;
	int bnt;
	while(!ifile.eof())
	{       pos1=ifile.tellg();
		ifile.read((char*)&m,sizeof(m));
		if(m.retmno()==mn)
		{       bnt= m.retbno();
			flag1=1;
			break;
		}
	}
	if(flag1==1)
	{
		ifile1.open("book.dat",ios::binary|ios::in|ios::out);
		while(!ifile1.eof())
		{	pos=ifile1.tellg();
			ifile1.read((char*)&b,sizeof(b));
			if(b.retbno()==bnt)
			{       flag2=1;
				int bt = b.rettot();
				int bi = b.retiss();
				if(bi<=bt)
				{       flag3=1;
					bi = bi-1;
					m.book = 0;
					b.issued=bi;
					ifile1.seekg(pos);
					ifile1.write((char*)&b,sizeof(b));
					ifile.seekg(pos1);
					ifile.write((char*)&m,sizeof(m));
				}
			}
		}
	}
	else
		cout<<"\nMember not available ";
	if(flag2==0)
		cout<<"\nBook not available";
	if(flag3==1)
		cout<<"\nBook returned ";
	else
		cout<<"\nCopies not available";
}
void modify()
{       clrscr();
	int ch;
	do
	{
		cout<<"\n MODIFY MENU ";
		cout<<"\n1.BOOK ";
		cout<<"\n2.MEMBER ";
		cout<<"\n3.BACK ";
		cout<<"\nEnter your choice ";
		cin>>ch;
		switch(ch)
		{case 1:clrscr();
			int bno,flag=0;
			 long pos;
			 ifstream bifile1;
			 book b;
			 bifile1.open("book.dat",ios::binary);
			cout<<"\nBNO"<<setw(7)<<"\t"<<"BNAME"<<setw(15)<<"AUTHOR"<<setw(15)<<"PUBLICATION"<<setw(15)<<"TOTAL"<<setw(5)<<"\tISSUED";
			while(bifile1.read((char*)&b,sizeof(b)))
			{
				b.show_book();
			}
			bifile1.close();
			cout<<"\n\n\nEnter the book no. to be modified ";
			cin>>bno;
			fstream bfile;
			bfile.open("book.dat",ios::binary|ios::in|ios::out);
			while(bfile)
			{	pos = bfile.tellg();
				bfile.read((char*)&b,sizeof(b));
				if(b.retbno()==bno)
				{       b.book_modify();
					bfile.seekg(pos);
					bfile.write((char*)&b,sizeof(b));
					flag = 1;
					break;
				}
			}
			if(flag==0)
				cout<<"\nBook does not exist ";
			bfile.close();
			ifstream bifile;
			bifile.open((char*)&b,sizeof(b));
			cout<<"\n\t\tAfter Modification ";
			bifile.open("book.dat",ios::binary);
			cout<<"\nBNO"<<setw(7)<<"\tBNAME"<<setw(15)<<"AUTHOR"<<setw(15)<<"PUBLICATION"<<setw(15)<<"TOTAL"<<setw(5)<<"\tISSUED";
			while(bifile.read((char*)&b,sizeof(b)))
			{
				b.show_book();
			}
			bifile.close();
			break;
		 case 2: clrscr();
			 int mno,flag1=0;
			 long pos1;
			 ifstream mifile1;
			 member m;
			 mifile1.open("member.dat",ios::binary);
			cout<<"\nMNO."<<setw(7)<<"\t"<<"MNAME"<<setw(15)<<"ADDRESS"<<setw(15)<<"PHONE"<<setw(15)<<"GENDER"<<setw(7)<<"\tBOOK NO.";
			while(mifile1.read((char*)&m,sizeof(m)))
			{	m.print_member();
			}
			mifile1.close();
			cout<<"\n\n\nEnter the member no. to be modified ";
			cin>>mno;
			fstream mfile;
			mfile.open("member.dat",ios::binary|ios::in|ios::out);
			while(mfile)
			{	pos1 = mfile.tellg();
				mfile.read((char*)&m,sizeof(m));
				if(m.retmno()==mno)
				{       m.member_modify();
					mfile.seekg(pos1);
					mfile.write((char*)&m,sizeof(m));
					flag1 = 1;
					break;
				}
			}
			if(flag1==0)
				cout<<"\nMember does not exist ";
			mfile.close();
			ifstream mifile;
			mifile.open((char*)&m,sizeof(m));
			cout<<"\n\t\tAfter Modification ";
			mifile.open("member.dat",ios::binary);
			cout<<"\nMNO."<<setw(7)<<"\t"<<"MNAME"<<setw(15)<<"ADDRESS"<<setw(15)<<"PHONE"<<setw(15)<<"GENDER"<<setw(7)<<"\tBOOK NO.";
			while(mifile.read((char*)&m,sizeof(m)))
			{	m.print_member();
			}
			mifile.close();
			break;
		 case 3: break;
	}
	}while(ch!=3);
}
void report_book()
{                               clrscr();
				int ch;
				book b;
				do
				{
					cout<<"\n\n\n1. All books ";
					cout<<"\n2. Specific book ";
					cout<<"\n3. Back";
					cout<<"\nEnter ur choice ";
					cin>>ch;
					switch(ch)
					{	case 1: clrscr();
							cout<<"\n\t\tBOOK DETAILS";
							ifstream ifile;
							ifile.open("book.dat",ios::binary);
							cout<<"\nBNO"<<setw(7)<<"\t"<<"BNAME"<<setw(15)<<"AUTHOR"<<setw(15)<<"PUBLICATION"<<setw(15)<<"TOTAL"<<setw(5)<<"\tISSUED";
							while(ifile.read((char*)&b,sizeof(b)))
							{
								b.show_book();
							}
							ifile.close();
							break;
					       case 2:  clrscr();
							cout<<"\n\t\tBOOK DETAILS";
							ifstream ifile1;
							int bno;
							ifile1.open("book.dat",ios::binary);
							cout<<"\nEnter book no.";
							cin>>bno;
							cout<<"\nBNO"<<setw(7)<<"\t"<<"BNAME"<<setw(15)<<"AUTHOR"<<setw(15)<<"PUBLICATION"<<setw(15)<<"TOTAL"<<setw(5)<<"\tISSUED";
							while(ifile1.read((char*)&b,sizeof(b)))
							{ if(b.retbno()==bno)
								b.show_book();
							}
							ifile.close();
							break;
					       case 3:  break;
					}
				}while(ch!=3);
}
void report_member()
{                               clrscr();
				int ch1;
				member m;
				do
				{
					cout<<"\n\n\n1. All members ";
					cout<<"\n2. Specific member ";
					cout<<"\n3. Back";
					cout<<"\nEnter ur choice ";
					cin>>ch1;
					switch(ch1)
					{	case 1: clrscr();
							cout<<"\n\t\tMEMBER DETAILS";
							ifstream mifile;
							mifile.open("member.dat",ios::binary);
							cout<<"\nMNO."<<setw(7)<<"\t"<<"MNAME"<<setw(15)<<"ADDRESS"<<setw(15)<<"PHONE"<<setw(15)<<"GENDER"<<setw(7)<<"\tBOOK NO.";
							while(mifile.read((char*)&m,sizeof(m)))
							{	m.print_member();
							}
							mifile.close();
							break;
					       case 2:  clrscr();
							cout<<"\n\t\tMEMBER DETAILS";
							ifstream mifile1;
							int mno;
							mifile1.open("member.dat",ios::binary);
							cout<<"\nEnter member no.";
							cin>>mno;
							cout<<"\nMNO."<<setw(7)<<"\t"<<"MNAME"<<setw(15)<<"ADDRESS"<<setw(15)<<"PHONE"<<setw(15)<<"GENDER"<<setw(7)<<"\tBOOK NO.";
							while(mifile1.read((char*)&m,sizeof(m)))
							{       if(m.retmno()==mno)
									m.print_member();
							}
							mifile1.close();
							break;
					       case 3:  break;
					}
				}while(ch1!=3);
}
void report()
{       clrscr();
	int ch;
	do
	{	cout<<"\n\n REPORT MENU ";
		cout<<"\n1.BOOK ";
		cout<<"\n2.MEMBER";
		cout<<"\n3.BACK";
		cout<<"\nEnter ur choice ";
		cin>>ch;
		switch(ch)
		{	case 1: report_book();
				break;
			case 2: report_member();
				break;
			case 3: break;
		}
	}while(ch!=3);
}
void add()
{       clrscr();
	int ch;
	do
	{
	cout<<"\n ADD MENU ";
	cout<<"\n1. BOOK ";
	cout<<"\n2. MEMBER ";
	cout<<"\n3. BACK ";
	cout<<"\nEnter your choice ";
	cin>>ch;
	switch(ch)
	{	case 1: write_book();
			break;
		case 2: write_member();
			break;
		case 3: break;
	}
	}while(ch!=3);
}
void show()
{       clrscr();
	int ch;
	book b;
	member m;
	do
	{
	cout<<"\n\n\n DISPLAY MENU ";
	cout<<"\n\n\t1.BOOK DETAILS ";
	cout<<"\n\t2.MEMBER DETILS ";
	cout<<"\n\t3.BACK";
	cout<<"\nEnter your choice ";
	cin>>ch;
	switch(ch)
	{
		case 1: clrscr();
			cout<<"\n\t\tBOOK DETAILS";
			ifstream ifile;
			ifile.open("book.dat",ios::binary);
			cout<<"\nBNO"<<setw(7)<<"\tBNAME"<<setw(15)<<"AUTHOR"<<setw(15)<<"PUBLICATION"<<setw(15)<<"TOTAL"<<setw(5)<<"\tISSUED";
			while(ifile.read((char*)&b,sizeof(b)))
			{
				b.show_book();
			}
			ifile.close();
			break;
		case 2: clrscr();
			cout<<"\n\t\tMEMBER DETAILS";
			ifstream ifile1;
			ifile1.open("member.dat",ios::binary);
			cout<<"\nMNO."<<setw(7)<<"\tMNAME"<<setw(15)<<"ADDRESS"<<setw(15)<<"PHONE"<<setw(15)<<"GENDER"<<setw(7)<<"\tBOOK NO.";
			while(ifile1.read((char*)&m,sizeof(m)))
			{	m.print_member();
			}
			ifile1.close();
			break;
		case 3: break;
	}
	}while(ch!=3);

}

void transaction()
{       clrscr();
	int ch;
	do
	{	cout<<"\n\n\nTransaction menu ";
		cout<<"\n1. Issue book";
		cout<<"\n2. Return book";
		cout<<"\n3. Back";
		cout<<"\nEnter ur choice ";
		cin>>ch;
		switch(ch)
		{	case 1: issue_book();
				break;
			case 2: return_book();
				break;
			case 3: break;
		}
	}while(ch!=3);
}
void main()
{	clrscr();
	int ch;
	do
	{       clrscr();
		cout<<"\n LIBRARY MANAGEMENT ";
		cout<<"\n1.ADD details";
		cout<<"\n2.SHOW details";
		cout<<"\n3.DELETE";
		cout<<"\n4.REPORT";
		cout<<"\n5.MODIFY";
		cout<<"\n6.TRANSACTION";
		cout<<"\n7.EXIT";
		cout<<"\nEnter your choice  ";
		cin>>ch;
		switch(ch)
		{	case 1: add();
				break;
			case 2: show();
				break;
			case 3: del();
				break;
			case 4: report();
				break;
			case 5: modify();
				break;
			case 6: transaction();
				break;
			case 7: exit(0);
		}
	}while(ch!=7);
	getch();
}

