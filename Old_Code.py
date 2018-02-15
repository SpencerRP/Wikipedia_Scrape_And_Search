def rake(title = 'Machine Learning', visited_categories = set()): #the first version of a recursive function, this suffered from rapid memory buildup.
    df = pd.DataFrame(query(title)); visited_categories.add(title);

    categories = set([title for title in df['title'].values if ('Category:' in title)])
    pages= set(df['title'].values) - categories
    
    categories = categories.difference(visited_categories)
    visited_categories.update(categories)

    if (len(categories) > 0):
        for title in categories:
            pages.update(rake(title, visited_categories = set()))
    return pages
visited_categories = set(); pages = set();
def rake2(title = 'Machine Learning'): #the second version of a recursive function, this edits the pages and visited_categories data structures, saving space.
    titles = pd.DataFrame(query(title))['title'].values; visited_categories.add(title);

    categories = set([title for title in titles if ('Category:' in title)]).difference(visited_categories)
    visited_categories.update(categories)
    pages.update(set(titles).difference(visted_categories))

    for title in categories:
        rake(title)
    return pages
def return_page_titles(): #an older (but more elegant) version of the function currently in use, this only returns a set of page titles.
    categories = set(str(x) for x in input().strip().split(',')); pages, visited = set(), set();
    while 1:
        if not len(categories):
            break
        visited.update(categories)
        try: titles = set(pd.DataFrame(query(categories.pop()))['title'].values)
        except: print(visited - categories)
        categories.update(set(title for title in titles if 'Category:' in title))
        pages.update(titles - categories);
    return pages

def recursive_search(name, origin = None, max_depth = 5): #Alexandra Brian's function, which we kind of collaborated on.
    if not origin: #If origin = None, i.e. original depth/search, set name as origin.
        origin = name #Honestly, the entire point of this is to enable laziness...
        
    df = pd.DataFrame(query(name))
    set_title = set(df['title'].values) #I have no idea why df.index works, but I'll do it differently.
    for cat in set_title:
        page_id = str(df.loc[cat ,'pageid'])
        #combined max_depth limit into same line as page-making limit. This way it still returns
        #the pages the df has. If you don't want this, put the max_depth limit at the top,
        #before you make the df/do the query.
        if ('Category:' in cat) & (max_depth > 0):
            recursive_search(cat, origin, max_depth - 1)
        else:
            text = contents_dict(page_id)
            page_dict = {'ultimate_category': origin, #the original search
                         'proximate_category': name, #the category that the page is under
                         'page_name': cat, #the name of the page
                         'page_id': page_id, #the id of the page
                         'text': text, #the text of the page
                        #'tfidf_vector' = tfidif(text),
                        }
                        #Does contents_dict clean the text? If it doesn't, it should.
                        #I recommend doing that and vectorizing the text at this stage, then
                        #putting the finished product into the db
                        #or at least that's what I'm doing, and I think it's sensible...
            col1_ref.insert_one(page_dict) #insert result into mongodb
            
            
#The Code Below This Isn't Actually Old/Deprecated, It's Just To Be Careful
def make_query(title):
    title = re.sub('\\s', '+', re.sub(':','%3A+', title.strip()))
    query = 'http://en.wikipedia.org/w/api.php?action=query&format=json&'+\
    'list=categorymembers&cmtitle={}'.format(title)+'&cmlimit=max'
    return query

def query(title):
    r = requests.get(make_query(title)).json()
    try:
        return r['query']['categorymembers']
    except:
        print('Error with: ', r)
        return {'title':[]}

def content_query(pageid):
    URL = 'http://en.wikipedia.org/w/api.php?action=query&format=json&titles={}&prop=extracts&rvprop=content'.format(pageid)
    query = requests.get(URL)
    return query

def get_text(pageid):
    try:
        r = content_query(pageid)
        text = r.json()['query']['pages'][pageid]['extract']
        return text
    except:
        print("{} is a problem with get_text".format(pageid))
        
def clean_text(text):
    text = re.findall('\u003E([\w\s\,\.][^\u003C]+)(?=\u003C[\w\/])', text)
    #find all text between ">" and "</".
    text = ' '.join(text) #make a string out of it
    text = unicodedata.normalize('NFKD', text) #take unicode like /xao3 and make them normal
    text = re.sub('\\n', '', re.sub('\s+', ' ', text)) #get rid of \n, and condense spaces.
    text = re.sub('[\.,](?=\w[^\.])', '. ', text) #Shark.b8 -> Shark. b8; B.C. -> B.C.
    
    text = re.sub("\'", "", text)
    #For some reason I can't simply replace \' with ', so I'm just getting rid of it all.
    #e.g. 'through Bayes\' rule.Sthrough Bayes\' rule.S'.replace("\'","'") works,
    #but if you do text = text.replace(...) it doesn't. I don't know why.
    
    text = re.sub(r"\B([A-Z])", r" \1", text) #or (?<=\w)([A-Z]). ABaby -> A Baby
    text = re.sub('(:)(?=\w)', ': ', text) #for example:child -> for example: child
    return text

def category_search(pickle = False): #Step 1
    original_categories = set(str(x) for x in input().strip().split(',')); 
    if pickle: #If I want to pickle something...:
        pickle_df = pd.DataFrame() 
        #Make an empty dataframe to use later in order to insert stuff into a dataframe.
        #So that I don't have to retrieve it from my MongoDB.
    else:
        client = pymongo.MongoClient('35.165.16.217', 27016)
        db_ref = client.wiki_database
        coll_ref = db_ref.wiki_text

    for original in original_categories:
        categories, visited = set(original), set();
        while 1:
            if not len(categories):
                break
            visited.update(categories)
            category = categories.pop()
            temp_df = pd.DataFrame(query(category))
            temp_df.set_index('title')
            try: titles = set(temp_df['title'].values)
            except: print(visited - categories)
            categories.update(set(title for title in titles if 'Category:' in title))
            for page in (titles - categories):
                page_id = temp_df.loc(page, 'pageid')
                temp_json = {'proximate_category': category,
                             'ultimate_category': original,
                             'page_name': page,
                             'page_id': page_id,
                             'text': clean_text(get_text(page_id))
                            }
                if pickle: 
                    pickle_df.concat(temp_json, axis = 0)
                else:
                    coll_ref.insert_one(temp_json)
    if pickle: 
        pickle_df.to_pickle('wiki_clean_text_df.p')
        
def page_match(search): #Step 2
    #Establish a dataframe from either a pickled version or pulled from MongoDB.
    try:
        df = pd.read_pickle('wiki_clean_text_df.p')
    except:
        try: 
            client = pymongo.MongoClient('35.165.16.217', 27016)
            db_ref = client.wiki_database
            coll_ref = db_ref.wiki_text #whatever your collection is called
            cursor = coll_ref.find()
            all_pages = list(cursor)
            df = pd.DataFrame(all_pages)
        except:
            print('Database inaccessible. Please try again when it is available.')
            return
    #Make a TFIDF Matrix from the database established above.
    tfidf_vectorizer = TfidfVectorizer(min_df = 2, stop_words = 'english')
    document_term_matrix_sps = tfidf_vectorizer.fit_transform(df.text)
    document_term_matrix_df = pd.DataFrame(document_term_matrix_sps.toarray(), 
                                           index=clean_data_df.page_name, 
                                           columns=tfidf_vectorizer.get_feature_names(),)
    
    try: 
        len(search) #if search > 1 string, the dataframe will have more than one row.
        keywords = [{'keyword':sample} for sample in list(search)]
    except: #if search = 1 string, the dataframe will have only one row.
        keywords = list([{'keyword':search}])
    
    #Establish a keyword dataframe from the "search" put in above.
    keywords_df = pd.DataFrame(keywords)
    keywords_encoded = tfidf_vectorizer.transform(keywords_df.keyword)
    keywords_encoded_df = pd.DataFrame(keyword_encoded.toarray(),
                             index = keywords_df.keyword,
                             columns=tfidf_vectorizer.get_feature_names())
    
    return_df = pd.DataFrame() #make an empty DataFrame
    
    for i in range(len(keywords)): #go through the search terms and do an SVD Cosine Similarity to them.
        random_keyword_df = keywords_encoded_df[i]
        dtm_with_search_term = document_term_matrix_df.append(keywords_encoded_df)
    
        #Make an SVD and apply it to dtm_with_search_term
        n_components = 50; SVD = TruncatedSVD(n_components)
        component_names = ['component_' + str(i+1) for i in range(n_components)]
        svd_matrix = SVD.fit_transform(dtm_with_search_term)
        svd_df = pd.DataFrame(svd_matrix, 
                              index=dtm_with_search_term.index,
                              columns=component_names)
    
        #Find and sort the cosine similarity for the search term.
        search_term_svd_vector = svd_df.loc[random_keyword_df.index]
        svd_df['cosine_sim'] = cosine_similarity(svd_df, search_term_svd_vector)
        
        #Take the five most correlated results and put them in a df w/ the initial search term.
        temp_df = svd_df[['cosine_sim']].sort_values('cosine_sim', ascending=False).head(5)
        temp_df['search'] = keywords[i] 
        return_df.concat(temp_df, axis=0) #Add it to the main dataframe we're returning.
        
    #Return a df w/ the 5 most correlated pages per search.
    return return_df
        
def category_recommendation(page_text):
    #Acquire a DataFrame made previously.
    try:
        df = pd.read_pickle('wiki_clean_text_df.p')
    except:
        try: 
            client = pymongo.MongoClient('35.165.16.217', 27016)
            db_ref = client.wiki_database
            coll_ref = db_ref.wiki_text #whatever your collection is called
            cursor = coll_ref.find()
            all_pages = list(cursor)
            df = pd.DataFrame(all_pages)
        except:
            print('Database inaccessible. Please try again when it is available.')
            return
    #Find the "nearest neighbors" i.e. those pages which have the highest correlation with
    #the target page using the function made for Step Two.
    nearest_neighbors = page_match(page_text)
    
    #Make a mask to get those pages from the df which are "neighbors" of the target page.
    is_a_neighbor = [title in nearest_neighbors.index for title in df['title'].values]
    neighbors = df['ultimate_category'][is_a_neighbor]
    
    #Make a dictionary with the count of each unique neighbor.
    #e.g., if we had 2 neighbors ML & BS, the dict might look like:
    #ML:3, BS:2
    neighbor_count = [{neighbor: neighbors.count(neighbor)} for neighbor in set(neighbors.values)]
    
    #Return the most common category possessed by the neighbors.
    #e.g., since ML has a count of 3, it will be returned.
    most_common = None
    for new in neighbor_count.keys():
        if most_common == None: 
            most_common = new
        if (neighbor_count[most_common] < neighbor_count[new]):
            most_common = new
    return most_common
    